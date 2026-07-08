"""JWT 기반 사용자 인증 (카카오 로그인 후 자체 토큰 발급/검증)"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import jwt
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config.settings import settings
from core.logging import logger

# auto_error=False: 비로그인(레거시 device_id) 요청도 허용해야 하는 엔드포인트가 있음
bearer_scheme = HTTPBearer(auto_error=False)

TOKEN_TYPE_ACCESS = "access"
TOKEN_TYPE_REFRESH = "refresh"


def _require_secret() -> str:
    """JWT 시크릿 키 확인 (미설정 시 인증 기능 비활성화 상태)"""
    if not settings.JWT_SECRET_KEY:
        logger.error("JWT_SECRET_KEY가 서버에 설정되지 않았습니다")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="로그인 서비스를 일시적으로 사용할 수 없습니다.",
        )
    return settings.JWT_SECRET_KEY


def _create_token(user_id: str, token_type: str, expires_delta: timedelta) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "type": token_type,
        "jti": uuid.uuid4().hex,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
    }
    return jwt.encode(payload, _require_secret(), algorithm=settings.JWT_ALGORITHM)


def create_access_token(user_id: str) -> str:
    """액세스 토큰 발급 (기본 1시간)"""
    return _create_token(
        user_id,
        TOKEN_TYPE_ACCESS,
        timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(user_id: str) -> str:
    """리프레시 토큰 발급 (기본 30일)"""
    return _create_token(
        user_id,
        TOKEN_TYPE_REFRESH,
        timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS),
    )


def decode_token(token: str, expected_type: str = TOKEN_TYPE_ACCESS) -> Dict[str, Any]:
    """
    토큰 검증 및 페이로드 반환

    Raises:
        HTTPException(401): 만료/변조/타입 불일치
    """
    try:
        payload = jwt.decode(
            token,
            _require_secret(),
            algorithms=[settings.JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="토큰이 만료되었습니다. 다시 로그인해주세요.",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 토큰입니다.",
        )

    if payload.get("type") != expected_type or not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 토큰입니다.",
        )

    return payload


async def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> str:
    """로그인 필수 엔드포인트용 의존성 - user_id 반환"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="로그인이 필요합니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(credentials.credentials, TOKEN_TYPE_ACCESS)
    return payload["sub"]


async def get_optional_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> Optional[str]:
    """
    로그인 선택 엔드포인트용 의존성 - 토큰 없으면 None (레거시 device_id 흐름 허용)

    토큰이 있는데 유효하지 않으면 401을 그대로 발생시킨다
    (만료된 토큰으로 무료 흐름을 타는 것을 방지).
    """
    if credentials is None:
        return None
    payload = decode_token(credentials.credentials, TOKEN_TYPE_ACCESS)
    return payload["sub"]
