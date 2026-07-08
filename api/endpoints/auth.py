"""회원 인증 엔드포인트 (카카오 로그인 + JWT)

로그인 흐름:
1. 앱이 카카오 SDK로 로그인 → 카카오 액세스 토큰 획득
2. POST /api/auth/kakao 로 토큰 전송
3. 서버가 카카오 API로 검증 → 회원 조회/생성 (신규 가입 시 보너스 크레딧)
4. 자체 JWT (access + refresh) 발급
"""

from typing import Optional

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
)
from database.user_repository import get_user_repository
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
            user = user_repo.create(
                kakao_id=profile["kakao_id"],
                nickname=profile["nickname"],
                email=profile.get("email"),
                initial_credits=0,
            )
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
        "access_token": create_access_token(user["user_id"]),
        "refresh_token": create_refresh_token(user["user_id"]),
        "token_type": "bearer",
        "is_new_user": is_new_user,
        "user": _public_user(user),
    }


@router.post("/auth/refresh")
@limiter.limit("20/minute")
async def refresh_token(request: Request, body: RefreshRequest):
    """리프레시 토큰으로 액세스 토큰 재발급"""
    payload = decode_token(body.refresh_token, TOKEN_TYPE_REFRESH)
    user_id = payload["sub"]

    return {
        "access_token": create_access_token(user_id),
        "token_type": "bearer",
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
