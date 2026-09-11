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
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from config.settings import settings
from core.logging import logger
from core.jwt_auth import (
    TOKEN_TYPE_REFRESH,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user_id,
    normalize_token_version,
    token_version_of,
)
from database.user_repository import (
    STATUS_ACTIVE,
    UserAlreadyExistsError,
    get_user_repository,
)
from services.credit_service import get_credit_service
from services.kakao_auth_service import get_kakao_auth_service

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


# ========== Request/Response Models ==========


class KakaoLoginRequest(BaseModel):
    kakao_access_token: str = Field(
        ..., min_length=1, description="카카오 SDK 액세스 토큰"
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


def _issue_tokens(user_id: str, token_version: int) -> dict:
    """액세스 + 리프레시 토큰 쌍 발급 (둘 다 tv 클레임 포함)"""
    return {
        "access_token": create_access_token(user_id, token_version),
        "refresh_token": create_refresh_token(user_id, token_version),
        "token_type": "bearer",
    }


def _public_user(user: dict) -> dict:
    """응답용 사용자 정보 (내부 필드 제외)"""
    return {
        "user_id": user["user_id"],
        "nickname": user.get("nickname"),
        "email": user.get("email"),
        "credits": int(user.get("credits", 0)),
        "training_consent": bool(user.get("training_consent", False)),
        "created_at": user.get("created_at"),
    }


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

    try:
        user_repo = get_user_repository()
        user = user_repo.get_by_kakao_id(profile["kakao_id"])

        is_new_user = user is None
        if is_new_user:
            try:
                user = user_repo.create(
                    kakao_id=profile["kakao_id"],
                    nickname=profile["nickname"],
                    email=profile.get("email"),
                    initial_credits=0,
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

        if is_new_user:
            if settings.SIGNUP_BONUS_CREDITS > 0:
                balance = get_credit_service().grant(
                    user["user_id"],
                    settings.SIGNUP_BONUS_CREDITS,
                    reason="signup_bonus",
                )
                user["credits"] = balance
        else:
            user_repo.update_last_login(user["user_id"])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 로그인 처리 실패: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="로그인 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        )

    return {
        **_issue_tokens(user["user_id"], _user_token_version(user)),
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

    return _issue_tokens(user_id, current_version)


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


@router.patch("/auth/me/consent")
async def update_consent(
    body: ConsentRequest, user_id: str = Depends(get_current_user_id)
):
    """
    원본 사진 AI 학습 활용 동의 변경 (선택 동의)

    동의 시: 이후 합성 요청의 원본 사진이 학습용으로 저장됨.
    철회 시: 이후 저장 중단 (기존 저장분 삭제는 별도 처리 필요).
    """
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
