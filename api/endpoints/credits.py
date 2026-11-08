"""크레딧 조회/구매 엔드포인트"""

import hashlib
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from config.settings import settings
from core.logging import log_structured, logger, mask_user_id
from core.jwt_auth import get_current_user_id, get_optional_user_id
from database.user_repository import get_user_repository
from services.free_policy_service import (
    REWARD_UNLOCK_FEATURES,
    hash_bucket,
    reward_grants_credit,
)
from services.admob_ssv_service import (
    InvalidSSVError,
    REWARD_MAX_AGE_SECONDS,
    SSVUnavailableError,
    get_admob_ssv_service,
    is_reward_timestamp_fresh,
    parse_ad_unit_allowlist,
)
from services.credit_service import RefAlreadyClaimedError, get_credit_service
from services.play_billing_service import (
    ACK_STATE_NOT_ACKNOWLEDGED,
    InvalidPurchaseError,
    PlayBillingUnavailableError,
    get_play_billing_service,
)
from services.usage_limit_service import get_usage_limit_service, validate_device_id

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# 계정당 한 번만 살 수 있는 상품 (첫 구매 할인팩)
ONE_PER_ACCOUNT_PRODUCTS = frozenset({"starter_5"})
# 결제 화면 기본 선택 상품
DEFAULT_PRODUCT_ID = "credits_10"
# 노출 A/B 대상 상품 ("추천" 배지 vs 보조 노출)
BADGE_AB_PRODUCT_ID = "credits_30"


class PurchaseRequest(BaseModel):
    product_id: str = Field(..., description="스토어 상품 ID (예: credits_10)")
    purchase_token: str = Field(..., description="Google Play 구매 토큰")


def _once_key(product_id: str, user_id: str) -> str:
    return f"once#{product_id}#{user_id}"


def _token_hash(purchase_token: str) -> str:
    return hashlib.sha256(purchase_token.encode("utf-8")).hexdigest()[:32]


def expected_obfuscated_account_id(user_id: str) -> str:
    """앱이 결제 때 setObfuscatedAccountId 로 넣어야 하는 값 = sha256(user_id) hex (64자)"""
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()


def _purchase_account_mismatch_response() -> JSONResponse:
    """다른 계정으로 결제된 영수증 403. 승인하지 않으므로 Google 이 3일 안에 자동 환불한다."""
    return JSONResponse(
        status_code=403,
        content={
            "error": "purchase_account_mismatch",
            "message": (
                "다른 계정으로 결제된 구매입니다. 결제한 계정으로 로그인해주세요. "
                "처리되지 않은 결제는 Google Play 에서 자동으로 환불됩니다."
            ),
        },
    )


class _OnePerAccountUsed(Exception):
    """계정당 1회 상품을 다른 구매로 이미 받은 계정"""


def _claim_one_per_account(
    credit_service, product_id: str, user_id: str, purchase_token: str
) -> Optional[str]:
    """계정당 1회 상품의 사용 마커 선점.

    같은 구매 토큰의 재시도는 통과시킨다 (토큰 중복 지급은 뒤의 purchase# 마커가 막는다).

    Returns:
        이번 요청이 새로 선점한 마커 키 (이후 실패 시 회수 대상).
        대상 상품이 아니거나 같은 토큰의 재시도면 None.

    Raises:
        _OnePerAccountUsed: 다른 구매로 이미 사용한 계정
    """
    if product_id not in ONE_PER_ACCOUNT_PRODUCTS:
        return None

    key = _once_key(product_id, user_id)
    token_hash = _token_hash(purchase_token)
    if credit_service.try_claim_ref(key, user_id, detail={"token_hash": token_hash}):
        return key

    marker = credit_service.get_claim(key) or {}
    if (marker.get("detail") or {}).get("token_hash") == token_hash:
        return None
    raise _OnePerAccountUsed()


def _starter_used_response(product_id: str) -> JSONResponse:
    """계정당 1회 상품 재구매 422.

    이 응답에서는 구매를 승인(acknowledge)하지 않으므로 Google 이 3일 안에 자동 환불한다.
    409 를 쓰면 안 된다 - 앱은 409 를 "이미 지급된 구매"로 보고 consume 하는데,
    consume 된 구매는 자동 환불되지 않는다.
    """
    return JSONResponse(
        status_code=422,
        content={
            "error": "one_per_account_used",
            "message": (
                "첫 구매 상품은 계정당 한 번만 구매할 수 있습니다. "
                "결제 금액은 Google Play 에서 자동으로 환불됩니다."
            ),
            "product_id": product_id,
        },
    )


@router.get("/credits/products")
async def list_credit_products(user_id: str = Depends(get_current_user_id)):
    """
    결제 화면용 상품 목록 + 잔액 + 광고 제거 여부

    가격은 내려주지 않는다 - 앱이 Play Billing ProductDetails 의 현지화 가격을 쓴다.

    Returns:
        {
            "products": [
                {"product_id": "starter_5", "credits": 5, "one_per_account": true,
                 "available": true, "default": false, "badge": null}, ...
            ],
            "paywall_variant": "recommended" | "secondary",  # credits_30 노출 A/B
            "balance": 3, "paid_balance": 0, "ad_free": false
        }
    """
    service = get_credit_service()
    try:
        wallet = service.get_wallet(user_id)
        once_used = {
            product_id: service.get_claim(_once_key(product_id, user_id)) is not None
            for product_id in ONE_PER_ACCOUNT_PRODUCTS
        }
    except ValueError:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"❌ 상품 목록 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="상품 정보를 불러오지 못했습니다.")

    variant = (
        "recommended"
        if hash_bucket(f"paywall-badge:{user_id}")
        < settings.PAYWALL_RECOMMENDED_BADGE_PERCENT
        else "secondary"
    )

    products = []
    for product_id, credits in settings.CREDIT_PRODUCTS.items():
        one_per_account = product_id in ONE_PER_ACCOUNT_PRODUCTS
        products.append(
            {
                "product_id": product_id,
                "credits": credits,
                "one_per_account": one_per_account,
                "available": not (one_per_account and once_used.get(product_id)),
                "default": product_id == DEFAULT_PRODUCT_ID,
                "badge": (
                    "recommended"
                    if product_id == BADGE_AB_PRODUCT_ID and variant == "recommended"
                    else None
                ),
            }
        )

    # 결제 화면 노출 분모 (paywall_view 는 앱 이벤트지만, 서버 쪽 교차 확인용)
    log_structured(
        "paywall_products_viewed",
        {"user_id": mask_user_id(user_id), "paywall_variant": variant},
    )
    return {
        "products": products,
        "paywall_variant": variant,
        "balance": wallet["balance"],
        "paid_balance": wallet["paid_balance"],
        "ad_free": bool(wallet["first_purchase_at"]),
    }


@router.get("/credits")
async def get_credits(user_id: str = Depends(get_current_user_id)):
    """크레딧 잔액 + 최근 증감 내역 조회

    has_purchased: 한 번이라도 결제했는지. first_purchase_at 이 기준이고, 그 속성이
    생기기 전에 결제한 회원은 최근 내역의 "purchase" 로 보완한다.
    """
    try:
        service = get_credit_service()
        wallet = service.get_wallet(user_id)
        history = service.get_history(user_id, limit=20)
    except ValueError:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"❌ 크레딧 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="크레딧 조회에 실패했습니다.")

    has_purchased = bool(wallet["first_purchase_at"]) or any(
        entry.get("reason") == "purchase" for entry in history
    )
    return {
        "balance": wallet["balance"],
        "history": history,
        "has_purchased": has_purchased,
    }


@router.post("/credits/purchase")
@limiter.limit("10/minute")
async def purchase_credits(
    request: Request,
    body: PurchaseRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    크레딧 구매 (Google Play 인앱결제 영수증 검증)

    1. purchases.products.get으로 purchase_token 검증
       + obfuscatedExternalAccountId 가 있으면 sha256(user_id) 와 일치해야 함 (불일치 403)
    2. 계정당 1회 상품(starter_5)이면 사용 마커 선점 - 이미 쓴 계정은 422 + 미승인
       (승인하지 않은 구매는 Google 이 3일 안에 자동 환불한다)
    3. 미승인 구매면 서버가 직접 acknowledge
       (미승인 구매는 3일 후 Google 자동 환불 → "환불 + 크레딧 유지" 차단)
    4. purchase_token 클레임 마커(조건부 put)로 중복 지급 방지
    5. credit_service.grant(reason="purchase", paid=True) - 지급 실패 시 마커 회수
    """
    amount = settings.CREDIT_PRODUCTS.get(body.product_id)
    if amount is None:
        raise HTTPException(status_code=400, detail="알 수 없는 상품입니다.")

    # 영수증 검증 (동기 requests 호출이라 스레드풀로 - 이벤트 루프 차단 방지)
    play_service = get_play_billing_service()
    try:
        receipt = await run_in_threadpool(
            play_service.verify_product_purchase, body.product_id, body.purchase_token
        )
    except InvalidPurchaseError:
        logger.warning(
            f"⚠️ 구매 영수증 검증 실패: user_id={user_id}, product={body.product_id}"
        )
        raise HTTPException(status_code=400, detail="구매 영수증 검증에 실패했습니다.")
    except PlayBillingUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="결제 검증 서비스를 일시적으로 사용할 수 없습니다.",
        )

    # 결제-계정 연결 확인: 결제 때 넣은 계정 값이 이 사용자와 다르면 거절.
    # 첫 구매 마커·승인보다 먼저 - 남의 영수증으로 마커를 선점하거나 승인시키지 못하게.
    # 값이 없는 결제(연결 기능 이전 앱)는 통과시킨다.
    account_id = receipt.get("obfuscated_account_id")
    if account_id and account_id != expected_obfuscated_account_id(user_id):
        logger.warning(
            f"⚠️ 결제 계정 불일치 거절: user_id={user_id}, product={body.product_id}"
        )
        log_structured(
            "purchase_account_mismatch",
            {"user_id": mask_user_id(user_id), "product_id": body.product_id},
        )
        return _purchase_account_mismatch_response()

    credit_service = get_credit_service()

    # 계정당 1회 상품: 승인 전에 확인해야 재구매분이 자동 환불된다
    try:
        once_key = _claim_one_per_account(
            credit_service, body.product_id, user_id, body.purchase_token
        )
    except _OnePerAccountUsed:
        logger.warning(
            f"⚠️ 계정당 1회 상품 재구매 거절: user_id={user_id}, product={body.product_id}"
        )
        return _starter_used_response(body.product_id)

    def release_once() -> None:
        if once_key:
            credit_service.release_ref(once_key)

    # 승인은 클레임 마커 생성 전에 수행 - 실패 시 503으로 재시도를 유도해도
    # 마커가 없으므로 다음 요청에서 지급까지 정상 재처리된다
    try:
        if receipt.get("acknowledgement_state") == ACK_STATE_NOT_ACKNOWLEDGED:
            await run_in_threadpool(
                play_service.acknowledge_product_purchase,
                body.product_id,
                body.purchase_token,
            )
    except InvalidPurchaseError:
        release_once()
        logger.warning(
            f"⚠️ 구매 승인 거부: user_id={user_id}, product={body.product_id}"
        )
        raise HTTPException(status_code=400, detail="구매 영수증 검증에 실패했습니다.")
    except PlayBillingUnavailableError:
        release_once()
        raise HTTPException(
            status_code=503,
            detail="결제 검증 서비스를 일시적으로 사용할 수 없습니다.",
        )

    # 같은 purchase_token 재사용 시 중복 지급 방지 (멱등성).
    # 마커·지급·원장을 한 트랜잭션으로 - 중간에 끊겨도 "마커만 있고 크레딧은 없는"
    # 상태가 생기지 않아, 재시도가 409 로 막혀 결제분을 잃는 일이 없다.
    ref_key = f"purchase#{body.purchase_token}"
    try:
        balance = credit_service.grant_with_claim(
            ref_key,
            user_id,
            amount,
            reason="purchase",
            ref_id=body.purchase_token,
            paid=True,
            detail={"product_id": body.product_id, "order_id": receipt.get("order_id")},
        )
    except RefAlreadyClaimedError:
        raise HTTPException(status_code=409, detail="이미 처리된 구매입니다.")
    except ValueError:
        release_once()
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    except Exception:
        # 트랜잭션이 반영되지 않았으므로 클라이언트가 그대로 재시도하면 된다
        release_once()
        logger.error(f"❌ 구매 크레딧 지급 실패: user_id={user_id}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="크레딧 지급 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )

    logger.info(
        f"💳 크레딧 구매 완료: user_id={user_id}, product={body.product_id}, "
        f"+{amount}, 잔액={balance}"
    )
    # purchase_complete (서버 검증 기준) - Phase 3 구매자 수·평균 결제액 집계용
    log_structured(
        "purchase_complete",
        {
            "user_id": mask_user_id(user_id),
            "product_id": body.product_id,
            "credits": amount,
        },
    )
    return {
        "success": True,
        "product_id": body.product_id,
        "granted": amount,
        "balance": balance,
        "order_id": receipt.get("order_id"),
    }


# 리워드 광고 1회 시청당 지급 크레딧
REWARD_AD_CREDIT = 1

# ad_unit 허용 목록 미설정 경고는 (개발/테스트 환경에서) 1회만 남긴다
_ad_unit_allowlist_warned = False


def _verify_ad_unit(ad_unit: Optional[str]) -> None:
    """콜백의 ad_unit이 우리 리워드 광고 단위인지 확인

    허용 목록이 비어 있으면 프로덕션에서는 fail-closed(503),
    개발/테스트에서는 경고만 남기고 통과시킨다.
    """
    global _ad_unit_allowlist_warned

    allowlist = parse_ad_unit_allowlist(settings.ADMOB_REWARD_AD_UNIT_IDS)
    if not allowlist:
        if settings.ENVIRONMENT == "production":
            logger.error(
                "❌ ADMOB_REWARD_AD_UNIT_IDS not configured "
                "- 리워드 콜백을 처리할 수 없습니다 (fail-closed)"
            )
            raise HTTPException(
                status_code=503,
                detail="검증 서비스를 일시적으로 사용할 수 없습니다.",
            )
        if not _ad_unit_allowlist_warned:
            logger.warning(
                "⚠️ ADMOB_REWARD_AD_UNIT_IDS 미설정 - ad_unit 검증을 건너뜁니다 "
                "(프로덕션에서는 거부됨)"
            )
            _ad_unit_allowlist_warned = True
        return

    if ad_unit not in allowlist:
        logger.warning(f"⚠️ 허용되지 않은 리워드 ad_unit: {ad_unit}")
        raise HTTPException(status_code=400, detail="검증에 실패했습니다.")


def _verify_timestamp(raw_timestamp: Optional[str]) -> None:
    """콜백 발급 시각 신선도 검사 (서명된 콜백의 무기한 재사용 방지)"""
    if raw_timestamp is None:
        logger.debug("SSV 콜백에 timestamp 없음 - 신선도 검사 생략")
        return

    if not is_reward_timestamp_fresh(raw_timestamp):
        logger.warning(
            f"⚠️ SSV 콜백 timestamp 만료/비정상: {raw_timestamp} "
            f"(허용 {REWARD_MAX_AGE_SECONDS}초)"
        )
        raise HTTPException(status_code=400, detail="검증에 실패했습니다.")


@router.get("/credits/reward-callback")
@limiter.limit("120/minute")
async def reward_ad_callback(request: Request):
    """
    AdMob 리워드 광고 SSV(서버측 검증) 콜백 - AdMob 서버가 호출

    앱이 아닌 AdMob이 호출하므로 JWT 대신 ECDSA 서명으로 인증한다.
    user_id는 앱이 광고 요청 시 ServerSideVerificationOptions로 설정한 값.
    - 회원: user_id 그대로
    - 비회원: "device:<device_id>" (기능 해제만 가능, 크레딧은 회원에게만)

    보상 방식 (_reward_grants_credit):
    - 크레딧 1 (대조군, REWARD_AD_GRANTS_CREDIT=True)
    - 원가 0원 기능 당일 해제 (실험군, 비회원, 또는 REWARD_AD_GRANTS_CREDIT=False)

    비정상(서명 불일치 등)은 4xx, 정상 처리됐지만 지급하지 않는 경우
    (중복 재시도/일일 상한)는 200 - AdMob이 non-200에 재시도하기 때문.
    """
    # 서명은 원본 쿼리 스트링 바이트에 대해 검증해야 함 (파싱/디코딩 전)
    raw_query = request.scope.get("query_string", b"")

    ssv_service = get_admob_ssv_service()
    try:
        params = await run_in_threadpool(ssv_service.verify_callback, raw_query)
    except InvalidSSVError:
        raise HTTPException(status_code=400, detail="검증에 실패했습니다.")
    except SSVUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="검증 서비스를 일시적으로 사용할 수 없습니다.",
        )

    # AdMob 콘솔의 "URL 확인" 핑은 서명은 유효하지만 user_id/transaction_id가 비어 있다.
    # 400을 주면 SSV 설정을 저장할 수 없으므로, 아무것도 지급하지 않고 200으로 답한다.
    # (서명 검증을 통과한 요청만 여기 도달하므로 위조로 악용될 수 없다.)
    user_id = params.get("user_id")
    transaction_id = params.get("transaction_id")
    if not user_id or not transaction_id:
        # 앱이 SSV 옵션에 user_id를 설정하지 않은 경우도 여기로 온다
        logger.info(
            "ℹ️ SSV 검증 핑 또는 user_id 없는 콜백 - 지급 없이 200 "
            f"(ad_unit={params.get('ad_unit')})"
        )
        return {"success": True, "rewarded": False, "reason": "verification_ping"}

    # 서명이 유효해도 우리 광고 단위가 아니거나 오래된 콜백이면 거부
    _verify_ad_unit(params.get("ad_unit"))
    _verify_timestamp(params.get("timestamp"))

    credit_service = get_credit_service()

    # 같은 transaction_id 재전송(AdMob 재시도 포함) 시 중복 지급 방지
    ref_key = f"reward#{transaction_id}"
    if not credit_service.try_claim_ref(
        ref_key, user_id, detail={"ad_unit": params.get("ad_unit")}
    ):
        return {"success": True, "rewarded": False, "reason": "duplicate"}

    # 유저당 일일 보상 상한 (조건부 원자 증가)
    try:
        under_limit = get_usage_limit_service().increment_daily_counter(
            f"reward_ad#{user_id}", settings.REWARD_AD_DAILY_LIMIT
        )
    except Exception:
        credit_service.release_ref(ref_key)
        logger.error("❌ 리워드 일일 카운터 갱신 실패", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="보상 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )

    if not under_limit:
        logger.info(f"⚠️ 리워드 일일 상한 도달: user_id={user_id}")
        return {"success": True, "rewarded": False, "reason": "daily_limit_reached"}

    try:
        unlock_subject = _unlock_subject_from_ssv(user_id)
        grants_credit = unlock_subject is None and _reward_grants_credit(user_id)
        if unlock_subject is None and not grants_credit:
            unlock_subject = f"user:{user_id}"
    except ValueError:
        # "device:" 뒤 형식 오류 - 재시도해도 같으므로 200
        credit_service.release_ref(ref_key)
        get_usage_limit_service().decrement_daily_counter(f"reward_ad#{user_id}")
        return {"success": True, "rewarded": False, "reason": "invalid_subject"}
    except Exception:
        credit_service.release_ref(ref_key)
        get_usage_limit_service().decrement_daily_counter(f"reward_ad#{user_id}")
        logger.error("❌ 리워드 보상 방식 확인 실패", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="보상 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )

    if not grants_credit:
        try:
            expires_at = get_usage_limit_service().set_daily_flag(
                f"unlock#{unlock_subject}"
            )
        except Exception:
            credit_service.release_ref(ref_key)
            get_usage_limit_service().decrement_daily_counter(f"reward_ad#{user_id}")
            logger.error("❌ 리워드 기능 해제 기록 실패", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="보상 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            )
        log_structured(
            "reward_unlock",
            {"subject": mask_user_id(unlock_subject), "expires_at": expires_at},
        )
        return {
            "success": True,
            "rewarded": True,
            "reward": "unlock",
            "features": list(REWARD_UNLOCK_FEATURES),
            "expires_at": expires_at,
        }

    try:
        balance = credit_service.grant(
            user_id, REWARD_AD_CREDIT, reason="reward_ad", ref_id=transaction_id
        )
    except ValueError:
        # 존재하지 않는 사용자 (탈퇴 등) - 재시도해도 소용없으므로 200
        credit_service.release_ref(ref_key)
        logger.warning(f"⚠️ 리워드 지급 대상 사용자 없음: user_id={user_id}")
        return {"success": True, "rewarded": False, "reason": "unknown_user"}
    except Exception:
        # 지급 실패 시 클레임 회수 + 일일 카운터 복구
        # (AdMob 재시도가 사용자 보상 한도를 소모하지 않도록) - 재처리되게 500
        credit_service.release_ref(ref_key)
        get_usage_limit_service().decrement_daily_counter(f"reward_ad#{user_id}")
        logger.error(f"❌ 리워드 크레딧 지급 실패: user_id={user_id}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="보상 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )

    logger.info(
        f"🎁 리워드 광고 크레딧 지급: user_id={user_id}, "
        f"+{REWARD_AD_CREDIT}, 잔액={balance}"
    )
    return {"success": True, "rewarded": True, "reward": "credit"}


DEVICE_SUBJECT_PREFIX = "device:"


def _unlock_subject_from_ssv(ssv_user_id: str) -> Optional[str]:
    """비회원 SSV user_id("device:<id>")면 해제 대상 키, 회원이면 None

    Raises:
        ValueError: device_id 형식 오류
    """
    if not ssv_user_id.startswith(DEVICE_SUBJECT_PREFIX):
        return None
    device_id = validate_device_id(ssv_user_id[len(DEVICE_SUBJECT_PREFIX) :])
    return f"device:{device_id}"


def _reward_grants_credit(user_id: str) -> bool:
    """회원의 보상형 광고 보상이 크레딧인지 (False 면 기능 해제)

    실험이 꺼져 있으면 사용자 조회 없이 전역 플래그만 본다.
    """
    if not settings.REWARD_AD_GRANTS_CREDIT:
        return False
    if not settings.FREE_POLICY_EXPERIMENT_ENABLED:
        return True
    user = get_user_repository().get_by_id(user_id)
    return reward_grants_credit((user or {}).get("free_policy_arm"))


@router.get("/rewards/unlocks")
async def get_reward_unlocks(
    device_id: Optional[str] = Query(None, description="비회원 디바이스 ID"),
    user_id: Optional[str] = Depends(get_optional_user_id),
):
    """
    보상형 광고로 해제된 원가 0원 기능 조회 (회원은 토큰, 비회원은 device_id)

    Returns:
        {
            "reward_mode": "credit" | "unlock",  # 지금 광고를 보면 받는 보상
            "unlocked": true,
            "features": ["personal_color_detail", "full_style_ranking"],
            "expires_at": 1730386800  # KST 자정 epoch 초 (해제 안 됐으면 null)
        }
    """
    if user_id:
        subject = f"user:{user_id}"
    elif device_id:
        try:
            subject = f"device:{validate_device_id(device_id)}"
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        raise HTTPException(
            status_code=400, detail="device_id 또는 로그인이 필요합니다."
        )

    try:
        expires_at = get_usage_limit_service().get_daily_flag(f"unlock#{subject}")
        grants_credit = bool(user_id) and _reward_grants_credit(user_id)
    except Exception as e:
        logger.error(f"❌ 보상 해제 상태 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="보상 정보를 불러오지 못했습니다.")

    return {
        "reward_mode": "credit" if grants_credit else "unlock",
        "unlocked": expires_at is not None,
        "features": list(REWARD_UNLOCK_FEATURES) if expires_at is not None else [],
        "expires_at": expires_at,
    }
