"""크레딧 조회/구매 엔드포인트"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from config.settings import settings
from core.logging import logger
from core.jwt_auth import get_current_user_id
from services.admob_ssv_service import (
    InvalidSSVError,
    SSVUnavailableError,
    get_admob_ssv_service,
)
from services.credit_service import get_credit_service
from services.play_billing_service import (
    InvalidPurchaseError,
    PlayBillingUnavailableError,
    get_play_billing_service,
)
from services.usage_limit_service import get_usage_limit_service

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


class PurchaseRequest(BaseModel):
    product_id: str = Field(..., description="스토어 상품 ID (예: credits_10)")
    purchase_token: str = Field(..., description="Google Play 구매 토큰")


@router.get("/credits")
async def get_credits(user_id: str = Depends(get_current_user_id)):
    """크레딧 잔액 + 최근 증감 내역 조회"""
    try:
        service = get_credit_service()
        balance = service.get_balance(user_id)
        history = service.get_history(user_id, limit=20)
    except ValueError:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"❌ 크레딧 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="크레딧 조회에 실패했습니다.")

    return {"balance": balance, "history": history}


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
    2. purchase_token 클레임 마커(조건부 put)로 중복 지급 방지
    3. credit_service.grant(reason="purchase") - 지급 실패 시 마커 회수
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

    # 같은 purchase_token 재사용 시 중복 지급 방지 (멱등성)
    credit_service = get_credit_service()
    ref_key = f"purchase#{body.purchase_token}"
    if not credit_service.try_claim_ref(
        ref_key,
        user_id,
        detail={"product_id": body.product_id, "order_id": receipt.get("order_id")},
    ):
        raise HTTPException(status_code=409, detail="이미 처리된 구매입니다.")

    try:
        balance = credit_service.grant(
            user_id, amount, reason="purchase", ref_id=body.purchase_token
        )
    except ValueError:
        credit_service.release_ref(ref_key)
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    except Exception:
        # 지급 실패 시 클레임 회수 - 클라이언트가 재시도할 수 있게
        credit_service.release_ref(ref_key)
        logger.error(f"❌ 구매 크레딧 지급 실패: user_id={user_id}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="크레딧 지급 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )

    logger.info(
        f"💳 크레딧 구매 완료: user_id={user_id}, product={body.product_id}, "
        f"+{amount}, 잔액={balance}"
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


@router.get("/credits/reward-callback")
@limiter.limit("120/minute")
async def reward_ad_callback(request: Request):
    """
    AdMob 리워드 광고 SSV(서버측 검증) 콜백 - AdMob 서버가 호출

    앱이 아닌 AdMob이 호출하므로 JWT 대신 ECDSA 서명으로 인증한다.
    user_id는 앱이 광고 요청 시 ServerSideVerificationOptions로 설정한 값.

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

    user_id = params.get("user_id")
    transaction_id = params.get("transaction_id")
    if not user_id or not transaction_id:
        # user_id는 앱에서 SSV 옵션으로 설정해야만 포함됨
        logger.warning("⚠️ SSV 콜백에 user_id/transaction_id 누락")
        raise HTTPException(status_code=400, detail="검증에 실패했습니다.")

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
        balance = credit_service.grant(
            user_id, REWARD_AD_CREDIT, reason="reward_ad", ref_id=transaction_id
        )
    except ValueError:
        # 존재하지 않는 사용자 (탈퇴 등) - 재시도해도 소용없으므로 200
        credit_service.release_ref(ref_key)
        logger.warning(f"⚠️ 리워드 지급 대상 사용자 없음: user_id={user_id}")
        return {"success": True, "rewarded": False, "reason": "unknown_user"}
    except Exception:
        # 지급 실패 시 클레임 회수 - AdMob 재시도에서 다시 처리되게 500
        credit_service.release_ref(ref_key)
        logger.error(f"❌ 리워드 크레딧 지급 실패: user_id={user_id}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="보상 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )

    logger.info(
        f"🎁 리워드 광고 크레딧 지급: user_id={user_id}, "
        f"+{REWARD_AD_CREDIT}, 잔액={balance}"
    )
    return {"success": True, "rewarded": True}
