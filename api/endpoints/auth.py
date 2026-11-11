"""회원 인증 엔드포인트 (카카오 로그인 + JWT)

로그인 흐름:
1. 앱이 카카오 SDK로 로그인 → 카카오 액세스 토큰 획득
2. POST /api/auth/kakao 로 토큰 전송
3. 서버가 카카오 API로 검증 → 회원 조회/생성 (신규 가입 시 보너스 크레딧)
4. 자체 JWT (access + refresh) 발급
"""

import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from config.settings import settings
from core.account_type import ACCOUNT_CUSTOMER, account_type_of_user, is_pro
from core.logging import log_structured, logger, mask_user_id
from core.jwt_auth import (
    TOKEN_TYPE_REFRESH,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_account_type,
    get_current_user_id,
    normalize_token_version,
    token_version_of,
)
from database.user_repository import (
    STATUS_ACTIVE,
    UserAlreadyExistsError,
    get_user_repository,
)
from services.credit_service import RefAlreadyClaimedError, get_credit_service
from services.free_policy_service import resolve_device_arm, signup_bonus_for
from services.kakao_auth_service import get_kakao_auth_service
from services.photo_storage_service import get_photo_storage_service
from services.pro_records_service import get_pro_records_service
from services.withdrawal_hold_service import hold_transact_item, recently_withdrawn
from services.usage_limit_service import get_usage_limit_service, validate_device_id

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


# ========== Request/Response Models ==========


class KakaoLoginRequest(BaseModel):
    kakao_access_token: str = Field(
        ..., min_length=1, description="카카오 SDK 액세스 토큰"
    )
    device_id: Optional[str] = Field(
        None,
        description="비로그인 때 쓰던 디바이스 ID (무료 정책 실험 코호트를 계정에 이어 붙인다)",
    )


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=1)


class ConsentRequest(BaseModel):
    training_consent: bool = Field(
        ..., description="원본 사진 AI 학습 활용 동의 (선택 동의)"
    )


# 동시 가입 경합 후 기존 계정을 찾을 때의 재시도 (GSI는 최종 일관성)
KAKAO_RACE_RETRY_ATTEMPTS = 3
KAKAO_RACE_RETRY_DELAY_SECONDS = 0.05


async def _resolve_raced_user(user_repo, kakao_id: str) -> Optional[Dict[str, Any]]:
    """동시 가입으로 create가 거부된 경우 먼저 생성된 계정을 조회

    GSI 조회는 최종 일관성이므로 몇 번 재시도하고, 그래도 없으면
    유일성 마커를 강한 일관성으로 읽어 실제 user_id를 얻는다.
    """
    for attempt in range(KAKAO_RACE_RETRY_ATTEMPTS):
        user = user_repo.get_by_kakao_id(kakao_id)
        if user is not None:
            return user
        await asyncio.sleep(KAKAO_RACE_RETRY_DELAY_SECONDS * (attempt + 1))

    return user_repo.get_by_kakao_marker(kakao_id)


def _user_token_version(user: dict) -> int:
    """사용자 아이템의 token_version (없으면 1 - 기존 회원 호환)"""
    return normalize_token_version(user.get("token_version"))


def _is_suspended(user: dict) -> bool:
    """계정 정지 여부 (status 속성이 없는 기존 회원은 정상 계정으로 취급)"""
    status_value = user.get("status")
    return status_value is not None and status_value != STATUS_ACTIVE


def _issue_tokens(
    user_id: str, token_version: int, account_type: str = ACCOUNT_CUSTOMER
) -> dict:
    """액세스 + 리프레시 토큰 쌍 발급 (둘 다 tv 클레임, Pro 는 acct 클레임 포함)"""
    return {
        "access_token": create_access_token(user_id, token_version, account_type),
        "refresh_token": create_refresh_token(user_id, token_version, account_type),
        "token_type": "bearer",
    }


def _account_type_mismatch_response() -> JSONResponse:
    """이미 있는 계정과 다른 앱으로 로그인한 경우 (손님 계정 ↔ Pro 계정 전환 금지)"""
    return JSONResponse(
        status_code=403,
        content={
            "error": "account_type_mismatch",
            "message": "이 카카오 계정은 다른 앱에서 가입되어 있어 로그인할 수 없습니다.",
        },
    )


def _public_user(user: dict) -> dict:
    """응답용 사용자 정보 (내부 필드 제외)

    ad_free: 한 번이라도 결제한 회원은 광고를 보여주지 않는다 (수익화 v2.1).
    """
    return {
        "user_id": user["user_id"],
        "nickname": user.get("nickname"),
        "email": user.get("email"),
        "credits": int(user.get("credits", 0)),
        "paid_credits": int(user.get("paid_credits", 0)),
        "ad_free": bool(user.get("first_purchase_at")),
        "training_consent": bool(user.get("training_consent", False)),
        "created_at": user.get("created_at"),
    }


def _signup_arm(device_id: Optional[str]) -> Optional[str]:
    """가입 기기의 무료 정책 코호트. device_id 가 없거나(구버전 앱) 형식이 틀리면 None"""
    if not device_id:
        return None
    try:
        validated = validate_device_id(device_id)
    except ValueError:
        return None
    return resolve_device_arm(get_usage_limit_service(), validated)


def _grant_pending_signup_bonus(user: dict, bonus: Optional[int] = None) -> int:
    """대기 중인 가입 보너스를 한 번만 지급하고 대기 표시를 지운다 (한 트랜잭션).

    가입 직후 지급이 실패해도 대기 표시가 남아 다음 로그인에서 다시 시도된다.
    마커(signup_bonus#<user_id>)가 있어 두 번 지급되지 않는다.

    Returns:
        이번에 지급한 보너스 (지급할 것이 없으면 0)
    """
    if bonus is None:
        bonus = int(user.get("signup_bonus_pending") or 0)
    if bonus <= 0:
        return 0
    try:
        balance = get_credit_service().grant_with_claim(
            f"signup_bonus#{user['user_id']}",
            user["user_id"],
            bonus,
            reason="signup_bonus",
            remove_attrs=("signup_bonus_pending",),
        )
    except RefAlreadyClaimedError:
        # 이미 지급됨 (대기 표시만 남은 비정상 상태) - 다시 주지 않는다
        logger.warning(
            f"⚠️ 가입 보너스 이미 지급됨: user_id={mask_user_id(user['user_id'])}"
        )
        return 0
    user["credits"] = balance
    user.pop("signup_bonus_pending", None)
    return bonus


# ========== Endpoints ==========


@router.post("/auth/kakao")
@limiter.limit("10/minute")
async def kakao_login(request: Request, body: KakaoLoginRequest):
    """
    카카오 로그인 (가입 겸용)

    신규 회원이면 자동 가입 + 보너스 크레딧 지급.

    Returns:
        {
            "access_token": "...",
            "refresh_token": "...",
            "token_type": "bearer",
            "is_new_user": bool,
            "user": { user_id, nickname, credits, ... }
        }
    """
    kakao_service = get_kakao_auth_service()
    profile = await kakao_service.verify_access_token(body.kakao_access_token)
    # 토큰을 발급한 카카오 앱으로 정해진 계정 유형 (클라이언트 입력이 아니다)
    account_type = profile.get("account_type") or ACCOUNT_CUSTOMER
    pro = is_pro(account_type)

    try:
        user_repo = get_user_repository()
        user = user_repo.get_by_kakao_id(profile["kakao_id"])

        is_new_user = user is None
        arm: Optional[str] = None
        pro_bonus = 0
        bonus_held = False
        if is_new_user:
            try:
                if pro:
                    # 탈퇴 후 보관 기간(약 6개월) 안에 다시 가입하면 체험을 주지 않는다
                    bonus_held = recently_withdrawn(profile["kakao_id"])
                    pro_bonus = 0 if bonus_held else settings.PRO_SIGNUP_BONUS_CREDITS
                    # Pro 계정은 무료 정책 실험에 넣지 않는다 (기기 코호트 등록도 하지 않음)
                    user = user_repo.create(
                        kakao_id=profile["kakao_id"],
                        nickname=profile["nickname"],
                        email=profile.get("email"),
                        initial_credits=0,
                        signup_bonus_pending=pro_bonus,
                        account_type=account_type,
                    )
                else:
                    arm = _signup_arm(body.device_id)
                    user = user_repo.create(
                        kakao_id=profile["kakao_id"],
                        nickname=profile["nickname"],
                        email=profile.get("email"),
                        initial_credits=0,
                        free_policy_arm=arm,
                        # 실험군 기기로 가입하면 보너스가 줄어든다 (free_policy_service 참고)
                        signup_bonus_pending=signup_bonus_for(arm),
                    )
            except UserAlreadyExistsError:
                # 동시 요청이 먼저 가입시킴 - 기존 로그인으로 전환 (보너스 없음)
                user = await _resolve_raced_user(user_repo, profile["kakao_id"])
                if user is None:
                    logger.error(
                        f"❌ 동시 가입 경합 후 기존 계정 조회 실패: "
                        f"kakao_id={profile['kakao_id']}"
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="로그인 처리 중 오류가 발생했습니다. "
                        "잠시 후 다시 시도해주세요.",
                    )
                is_new_user = False

        if _is_suspended(user):
            logger.warning(f"⛔ 정지 계정 로그인 시도: user_id={user['user_id']}")
            raise HTTPException(
                status_code=403,
                detail="이용이 정지된 계정입니다. 고객센터에 문의해주세요.",
            )

        # 손님 계정을 Pro 앱으로(또는 반대로) 쓰지 못하게 한다.
        # 카카오 회원번호는 앱마다 따로 나오므로 보통은 생기지 않는다.
        if account_type_of_user(user) != account_type:
            logger.warning(
                f"⛔ 계정 유형 불일치 로그인 거절: user_id={mask_user_id(user['user_id'])}, "
                f"login={account_type}"
            )
            return _account_type_mismatch_response()

        if is_new_user:
            if pro:
                bonus = _grant_pending_signup_bonus(user, pro_bonus)
                log_structured(
                    "signup",
                    {
                        "user_id": mask_user_id(user["user_id"]),
                        "arm": None,
                        "bonus": bonus,
                        "account_type": account_type,
                        "bonus_held": bonus_held,
                    },
                )
            else:
                bonus = _grant_pending_signup_bonus(user, signup_bonus_for(arm))
                log_structured(
                    "signup",
                    {
                        "user_id": mask_user_id(user["user_id"]),
                        "arm": arm,
                        "bonus": bonus,
                    },
                )
        else:
            user_repo.update_last_login(user["user_id"])
            # 가입 때 보너스 지급이 실패한 계정이면 지금 다시 지급.
            # 여기서 또 실패해도 로그인은 막지 않는다 (대기 표시가 남아 다음에 재시도)
            if user.get("signup_bonus_pending"):
                try:
                    _grant_pending_signup_bonus(user)
                except Exception:
                    logger.error(
                        "❌ 대기 가입 보너스 재지급 실패: "
                        f"user_id={mask_user_id(user['user_id'])}",
                        exc_info=True,
                    )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 로그인 처리 실패: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="로그인 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        )

    return {
        **_issue_tokens(user["user_id"], _user_token_version(user), account_type),
        "is_new_user": is_new_user,
        "user": _public_user(user),
    }


@router.post("/auth/refresh")
@limiter.limit("20/minute")
async def refresh_token(request: Request, body: RefreshRequest):
    """리프레시 토큰으로 액세스/리프레시 토큰 재발급

    액세스 토큰 검증과 달리 여기서는 반드시 사용자 아이템을 강한 일관성으로
    읽어 (1) 계정 존재 (2) status == active (3) tv 클레임 == 사용자의
    token_version 을 확인한다. 강제 로그아웃(logout-all)/계정 정지는 이 경로에서
    즉시 반영된다 (이미 발급된 액세스 토큰은 만료까지 최대 60분 유효).
    """
    payload = decode_token(body.refresh_token, TOKEN_TYPE_REFRESH)
    user_id = payload["sub"]
    claimed_version = token_version_of(payload)

    try:
        user = get_user_repository().get_by_id_consistent(user_id)
    except Exception as e:
        logger.error(f"❌ 리프레시 사용자 조회 실패: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="로그인 서비스에 일시적 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        )

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="유효하지 않은 토큰입니다. 다시 로그인해주세요.",
        )

    if _is_suspended(user):
        logger.warning(f"⛔ 정지 계정 리프레시 시도: user_id={user_id}")
        raise HTTPException(
            status_code=401,
            detail="이용이 정지된 계정입니다. 고객센터에 문의해주세요.",
        )

    current_version = _user_token_version(user)
    if claimed_version != current_version:
        logger.warning(
            f"⛔ 무효화된 리프레시 토큰: user_id={user_id}, "
            f"tv={claimed_version} != {current_version}"
        )
        raise HTTPException(
            status_code=401,
            detail="만료된 세션입니다. 다시 로그인해주세요.",
        )

    # 계정 유형은 회원 행 기준 (토큰 클레임이 아니라) - 갱신 후에도 Pro 가 유지된다
    return _issue_tokens(user_id, current_version, account_type_of_user(user))


@router.post("/auth/logout-all")
@limiter.limit("10/minute")
async def logout_all(request: Request, user_id: str = Depends(get_current_user_id)):
    """모든 기기에서 로그아웃 (token_version 증가 → 리프레시 토큰 전부 무효화)

    이미 발급된 액세스 토큰은 만료(기본 60분)까지 유효하다.
    """
    try:
        new_version = get_user_repository().bump_token_version(user_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"❌ 전체 로그아웃 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="로그아웃 처리에 실패했습니다.")

    logger.info(f"🔐 전체 로그아웃: user_id={user_id}, token_version={new_version}")
    return {
        "success": True,
        "token_version": new_version,
        "message": "모든 기기에서 로그아웃되었습니다. 다시 로그인해주세요.",
    }


@router.get("/auth/me")
async def get_me(user_id: str = Depends(get_current_user_id)):
    """내 프로필 + 크레딧 잔액 조회"""
    try:
        user = get_user_repository().get_by_id(user_id)
    except Exception as e:
        logger.error(f"❌ 프로필 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="프로필 조회에 실패했습니다.")

    if user is None:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    return {"user": _public_user(user)}


def _withdrawal_error(status_code: int, error: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"error": error, "message": message}
    )


@router.delete("/auth/me")
@limiter.limit("5/minute")
async def withdraw(
    request: Request,
    user_id: str = Depends(get_current_user_id),
    account_type: str = Depends(get_account_type),
):
    """
    회원 탈퇴 (지금은 Pro 계정만)

    1. pro/results/{user_id}/ 의 결과 사진과 공유 주소 연결 정보를 지운다.
       실패하면 503 으로 멈춘다 (계정이 남아 있어야 다시 시도할 수 있다).
    2. 손님 기록(services/pro_records_service.py)을 전부 지운다: pro/records/{user_id}/ 사진 →
       손님 기록 테이블의 이 디자이너 파티션. 실패하면 503 (위와 같은 이유).
       테이블이 설정되지 않았거나 아직 없으면 지울 것이 없으므로 건너뛴다.
    3. 회원 행과 카카오 유일성 마커를 지운다. 남은 합성 횟수는 사라진다.
       크레딧 원장·결제 마커는 결제 기록 보관을 위해 남긴다 (user_id 만 있음).
       같은 트랜잭션에서 카카오 회원번호를 되돌릴 수 없게 바꾼 값을 약 6개월 보관한다
       (그 기간의 재가입에는 체험을 주지 않는다 - services/withdrawal_hold_service.py).
    이미 탈퇴한 계정의 재요청(같은 액세스 토큰)은 결과 폴더·손님 기록만 다시 확인하고 성공으로 답한다.

    손님 계정은 403 withdrawal_not_supported (이번 범위 밖).
    카카오 연결 끊기(unlink)는 앱이 카카오 SDK 로 한다.

    Returns:
        {"success": true, "deleted_results": int, "deleted_records": int(지운 손님 수)}
    """
    not_supported = _withdrawal_error(
        403,
        "withdrawal_not_supported",
        "이 앱에서는 탈퇴를 지원하지 않는 계정입니다. 고객센터로 문의해주세요.",
    )
    if not is_pro(account_type):
        return not_supported

    user_repo = get_user_repository()
    try:
        user = user_repo.get_by_id_consistent(user_id)
    except Exception as e:
        logger.error(f"❌ 탈퇴 사용자 조회 실패: {str(e)}")
        return _withdrawal_error(
            503,
            "withdrawal_failed",
            "탈퇴 처리에 실패했습니다. 잠시 후 다시 시도해주세요.",
        )
    if user is not None and not is_pro(account_type_of_user(user)):
        return not_supported

    try:
        deleted_results = await run_in_threadpool(
            get_photo_storage_service().delete_user_results, user_id, True
        )
    except Exception as e:
        logger.error(
            f"❌ 탈퇴 결과 사진 삭제 실패: user_id={mask_user_id(user_id)}, {str(e)}"
        )
        return _withdrawal_error(
            503,
            "withdrawal_failed",
            "탈퇴 처리에 실패했습니다. 잠시 후 다시 시도해주세요.",
        )

    try:
        deleted_records = await run_in_threadpool(
            get_pro_records_service().delete_all_records, user_id
        )
    except Exception as e:
        logger.error(
            f"❌ 탈퇴 손님 기록 삭제 실패: user_id={mask_user_id(user_id)}, "
            f"{type(e).__name__}"
        )
        return _withdrawal_error(
            503,
            "withdrawal_failed",
            "탈퇴 처리에 실패했습니다. 잠시 후 다시 시도해주세요.",
        )

    if user is not None:
        try:
            user_repo.delete_account(
                user_id,
                user["kakao_id"],
                extra_items=[hold_transact_item(user["kakao_id"])],
            )
        except ValueError:
            pass  # 동시 요청이 먼저 지움
        except Exception as e:
            logger.error(
                f"❌ 탈퇴 회원 삭제 실패: user_id={mask_user_id(user_id)}, {str(e)}"
            )
            return _withdrawal_error(
                503,
                "withdrawal_failed",
                "탈퇴 처리에 실패했습니다. 잠시 후 다시 시도해주세요.",
            )

    log_structured(
        "account_withdrawn",
        {
            "user_id": mask_user_id(user_id),
            "account_type": account_type,
            "deleted_results": deleted_results,
            "deleted_records": deleted_records,
            "already_deleted": user is None,
        },
    )
    return {
        "success": True,
        "deleted_results": deleted_results,
        "deleted_records": deleted_records,
    }


@router.patch("/auth/me/consent")
async def update_consent(
    body: ConsentRequest,
    user_id: str = Depends(get_current_user_id),
    account_type: str = Depends(get_account_type),
):
    """
    원본 사진 AI 학습 활용 동의 변경 (선택 동의)

    동의 시: 이후 합성 요청의 원본 사진이 학습용으로 저장됨.
    철회 시: 이후 저장 중단 (기존 저장분 삭제는 별도 처리 필요).

    Pro 계정은 403: 사진 속 사람이 계정 주인이 아니라 손님이므로 동의를 대신할 수 없다.
    """
    if is_pro(account_type):
        return JSONResponse(
            status_code=403,
            content={
                "error": "pro_consent_not_allowed",
                "message": "이 계정은 사진을 학습에 쓰지 않습니다.",
            },
        )
    try:
        get_user_repository().set_training_consent(user_id, body.training_consent)
    except ValueError:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"❌ 동의 변경 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="동의 설정 변경에 실패했습니다.")

    return {
        "success": True,
        "training_consent": body.training_consent,
    }
