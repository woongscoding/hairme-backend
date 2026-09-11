"""JWT 인증 모듈 테스트"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

import time
from datetime import timedelta

import jwt as pyjwt
import pytest
from fastapi import HTTPException

from config.settings import settings
from core.jwt_auth import (
    DEFAULT_TOKEN_VERSION,
    TOKEN_TYPE_ACCESS,
    TOKEN_TYPE_REFRESH,
    _create_token,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user_id,
    get_optional_user_id,
    normalize_token_version,
    token_version_of,
)


class TestTokenCreation:
    def test_access_token_roundtrip(self):
        token = create_access_token("user-123")
        payload = decode_token(token, TOKEN_TYPE_ACCESS)

        assert payload["sub"] == "user-123"
        assert payload["type"] == TOKEN_TYPE_ACCESS
        assert payload["exp"] > payload["iat"]

    def test_refresh_token_roundtrip(self):
        token = create_refresh_token("user-456")
        payload = decode_token(token, TOKEN_TYPE_REFRESH)

        assert payload["sub"] == "user-456"
        assert payload["type"] == TOKEN_TYPE_REFRESH

    def test_tokens_have_unique_jti(self):
        t1 = create_access_token("user-1")
        t2 = create_access_token("user-1")
        p1 = decode_token(t1)
        p2 = decode_token(t2)
        assert p1["jti"] != p2["jti"]


class TestTokenVersionClaim:
    """tv(token_version) 클레임 - 리프레시 토큰 무효화(강제 로그아웃/정지)용"""

    def test_access_and_refresh_tokens_carry_tv(self):
        access = decode_token(create_access_token("user-1", 7), TOKEN_TYPE_ACCESS)
        refresh = decode_token(create_refresh_token("user-1", 7), TOKEN_TYPE_REFRESH)

        assert access["tv"] == 7
        assert refresh["tv"] == 7

    def test_default_token_version_is_one(self):
        payload = decode_token(create_access_token("user-1"))
        assert payload["tv"] == DEFAULT_TOKEN_VERSION == 1

    def test_token_version_of_missing_claim_is_one(self):
        """tv 클레임이 없는 과거 토큰은 1로 간주 (기존 세션 유지)"""
        assert token_version_of({"sub": "user-1"}) == 1

    def test_normalize_token_version(self):
        assert normalize_token_version(None) == 1
        assert normalize_token_version("3") == 3
        assert normalize_token_version(5) == 5
        assert normalize_token_version("bogus") == 1


class TestTokenValidation:
    def test_refresh_token_rejected_as_access(self):
        """리프레시 토큰으로 API 접근 불가 (타입 검증)"""
        token = create_refresh_token("user-123")
        with pytest.raises(HTTPException) as exc_info:
            decode_token(token, TOKEN_TYPE_ACCESS)
        assert exc_info.value.status_code == 401

    def test_expired_token_rejected(self):
        token = _create_token("user-123", TOKEN_TYPE_ACCESS, timedelta(seconds=-10))
        with pytest.raises(HTTPException) as exc_info:
            decode_token(token, TOKEN_TYPE_ACCESS)
        assert exc_info.value.status_code == 401
        assert "만료" in exc_info.value.detail

    def test_tampered_token_rejected(self):
        """다른 시크릿으로 서명된 토큰 거부"""
        forged = pyjwt.encode(
            {"sub": "attacker", "type": "access", "exp": int(time.time()) + 3600},
            "wrong_secret_key",
            algorithm=settings.JWT_ALGORITHM,
        )
        with pytest.raises(HTTPException) as exc_info:
            decode_token(forged, TOKEN_TYPE_ACCESS)
        assert exc_info.value.status_code == 401

    def test_garbage_token_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            decode_token("not.a.jwt", TOKEN_TYPE_ACCESS)
        assert exc_info.value.status_code == 401


class TestDependencies:
    @pytest.mark.asyncio
    async def test_required_auth_without_credentials(self):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_id(credentials=None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_optional_auth_without_credentials_returns_none(self):
        """토큰 없으면 None (레거시 device_id 흐름 허용)"""
        result = await get_optional_user_id(credentials=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_optional_auth_with_invalid_token_raises(self):
        """만료/변조 토큰은 None이 아니라 401 (무료 흐름 우회 방지)"""
        from fastapi.security import HTTPAuthorizationCredentials

        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="bad.token")
        with pytest.raises(HTTPException):
            await get_optional_user_id(credentials=creds)
