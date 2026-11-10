"""JWT 기반 사용자 인증 (카카오 로그인 후 자체 토큰 발급/검증)"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import jwt
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config.settings import settings
from core.account_type import (
    ACCOUNT_CUSTOMER,
    ACCOUNT_TYPE_CLAIM,
    account_type_of_payload,
    is_pro,
)
from core.logging import logger

# auto_error=False: 비로그인(레거시 device_id) 요청도 허용해야 하는 엔드포인트가 있음
bearer_scheme = HTTPBearer(auto_error=False)

TOKEN_TYPE_ACCESS = "access"
TOKEN_TYPE_REFRESH = "refresh"

# tv(token_version) 클레임이 없는 토큰(이 기능 도입 전 발급분)은 1로 간주한다.
# 사용자 아이템에 token_version 속성이 없는 기존 회원도 동일하게 1이다.
DEFAULT_TOKEN_VERSION = 1


def _require_secret() -> str:
    """JWT 시크릿 키 확인 (미설정 시 인증 기능 비활성화 상태)"""
    if not settings.JWT_SECRET_KEY:
        logger.error("JWT_SECRET_KEY가 서버에 설정되지 않았습니다")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="로그인 서비스를 일시적으로 사용할 수 없습니다.",
        )
    return settings.JWT_SECRET_KEY


def normalize_token_version(value: Any) -> int:
    """token_version 값을 정수로 정규화 (없거나 이상하면 DEFAULT_TOKEN_VERSION)"""
    if value is None:
        return DEFAULT_TOKEN_VERSION
    try:
        return int(value)
    except (TypeError, ValueError):
        return DEFAULT_TOKEN_VERSION


def _create_token(
    user_id: str,
    token_type: str,
    expires_delta: timedelta,
    token_version: int = DEFAULT_TOKEN_VERSION,
    account_type: Optional[str] = None,
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "type": token_type,
        "tv": normalize_token_version(token_version),
        "jti": uuid.uuid4().hex,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
    }
    # Pro 계정만 클레임을 넣는다. 손님 토큰은 기존과 같은 클레임 구성을 유지한다.
    if is_pro(account_type):
        payload[ACCOUNT_TYPE_CLAIM] = account_type
    return jwt.encode(payload, _require_secret(), algorithm=settings.JWT_ALGORITHM)


def create_access_token(
    user_id: str,
    token_version: int = DEFAULT_TOKEN_VERSION,
    account_type: Optional[str] = None,
) -> str:
    """액세스 토큰 발급 (기본 1시간)"""
    return _create_token(
        user_id,
        TOKEN_TYPE_ACCESS,
        timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
        token_version=token_version,
        account_type=account_type,
    )


def create_refresh_token(
    user_id: str,
    token_version: int = DEFAULT_TOKEN_VERSION,
    account_type: Optional[str] = None,
) -> str:
    """리프레시 토큰 발급 (기본 30일)"""
    return _create_token(
        user_id,
        TOKEN_TYPE_REFRESH,
        timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS),
        token_version=token_version,
        account_type=account_type,
    )


def token_version_of(payload: Dict[str, Any]) -> int:
    """토큰 페이로드의 tv 클레임 (없으면 DEFAULT_TOKEN_VERSION)"""
    return normalize_token_version(payload.get("tv"))


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
    """로그인 필수 엔드포인트용 의존성 - user_id 반환

    액세스 토큰 검증은 의도적으로 stateless 다 (DB 조회 없음).
    token_version 대조는 리프레시(POST /api/auth/refresh)에서만 수행하므로,
    강제 로그아웃/정지는 액세스 토큰 만료(기본 60분) 후 완전히 적용된다.
    """
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


async def get_account_type(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> str:
    """요청한 계정의 유형 (DB 조회 없이 액세스 토큰의 acct 클레임으로 판단)

    토큰이 없거나 검증에 실패하면 손님으로 본다. 검증 실패를 여기서 401 로 바꾸지
    않는 이유: 이 의존성은 항상 get_current_user_id / get_optional_user_id 와 함께
    쓰이고, 그쪽이 같은 토큰으로 401 을 낸다. 따라서 "Pro 토큰인데 손님 처리" 가
    실제로 실행되는 경로는 없다.
    """
    if credentials is None:
        return ACCOUNT_CUSTOMER
    try:
        payload = decode_token(credentials.credentials, TOKEN_TYPE_ACCESS)
    except HTTPException:
        return ACCOUNT_CUSTOMER
    return account_type_of_payload(payload)
