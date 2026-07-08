"""회원 인증 엔드포인트 테스트 (카카오 API + DynamoDB 모킹)"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from core.jwt_auth import create_access_token, create_refresh_token
from main import app

KAKAO_PROFILE = {
    "kakao_id": "12345678",
    "nickname": "테스트유저",
    "email": "test@example.com",
}

EXISTING_USER = {
    "user_id": "existing-user-id",
    "kakao_id": "12345678",
    "nickname": "테스트유저",
    "email": "test@example.com",
    "credits": 3,
    "training_consent": False,
    "status": "active",
    "created_at": "2026-07-01T00:00:00+00:00",
}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_kakao():
    """카카오 토큰 검증을 모킹 (외부 API 호출 방지)"""
    service = MagicMock()
    service.verify_access_token = AsyncMock(return_value=dict(KAKAO_PROFILE))
    with patch("api.endpoints.auth.get_kakao_auth_service", return_value=service):
        yield service


class TestKakaoLogin:
    def test_login_existing_user(self, client, mock_kakao):
        repo = MagicMock()
        repo.get_by_kakao_id.return_value = dict(EXISTING_USER)

        with patch("api.endpoints.auth.get_user_repository", return_value=repo):
            response = client.post(
                "/api/auth/kakao", json={"kakao_access_token": "valid_kakao_token"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["is_new_user"] is False
        assert data["access_token"]
        assert data["refresh_token"]
        assert data["user"]["user_id"] == "existing-user-id"
        assert data["user"]["credits"] == 3
        repo.update_last_login.assert_called_once_with("existing-user-id")

    def test_login_new_user_gets_signup_bonus(self, client, mock_kakao):
        new_user = dict(EXISTING_USER, user_id="new-user-id", credits=0)
        repo = MagicMock()
        repo.get_by_kakao_id.return_value = None
        repo.create.return_value = new_user

        credit_service = MagicMock()
        credit_service.grant.return_value = 5

        with patch("api.endpoints.auth.get_user_repository", return_value=repo), patch(
            "api.endpoints.auth.get_credit_service", return_value=credit_service
        ):
            response = client.post(
                "/api/auth/kakao", json={"kakao_access_token": "valid_kakao_token"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["is_new_user"] is True
        assert data["user"]["credits"] == 5
        credit_service.grant.assert_called_once()
        assert credit_service.grant.call_args.kwargs["reason"] == "signup_bonus"

    def test_login_with_invalid_kakao_token(self, client):
        from fastapi import HTTPException

        service = MagicMock()
        service.verify_access_token = AsyncMock(
            side_effect=HTTPException(status_code=401, detail="카카오 인증 실패")
        )

        with patch("api.endpoints.auth.get_kakao_auth_service", return_value=service):
            response = client.post(
                "/api/auth/kakao", json={"kakao_access_token": "bad_token"}
            )

        assert response.status_code == 401

    def test_login_missing_token(self, client):
        response = client.post("/api/auth/kakao", json={})
        assert response.status_code == 422


class TestRefresh:
    def test_refresh_returns_new_access_token(self, client):
        refresh = create_refresh_token("user-123")
        response = client.post("/api/auth/refresh", json={"refresh_token": refresh})

        assert response.status_code == 200
        assert response.json()["access_token"]

    def test_access_token_rejected_for_refresh(self, client):
        """액세스 토큰으로는 재발급 불가"""
        access = create_access_token("user-123")
        response = client.post("/api/auth/refresh", json={"refresh_token": access})
        assert response.status_code == 401


class TestMe:
    def test_me_requires_auth(self, client):
        response = client.get("/api/auth/me")
        assert response.status_code == 401

    def test_me_returns_profile(self, client):
        repo = MagicMock()
        repo.get_by_id.return_value = dict(EXISTING_USER)
        token = create_access_token("existing-user-id")

        with patch("api.endpoints.auth.get_user_repository", return_value=repo):
            response = client.get(
                "/api/auth/me", headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        user = response.json()["user"]
        assert user["user_id"] == "existing-user-id"
        assert user["training_consent"] is False
        # 내부 필드는 응답에 노출되지 않음
        assert "kakao_id" not in user

    def test_me_with_expired_style_bad_token(self, client):
        response = client.get(
            "/api/auth/me", headers={"Authorization": "Bearer invalid.token.here"}
        )
        assert response.status_code == 401


class TestConsent:
    def test_update_consent(self, client):
        repo = MagicMock()
        token = create_access_token("existing-user-id")

        with patch("api.endpoints.auth.get_user_repository", return_value=repo):
            response = client.patch(
                "/api/auth/me/consent",
                json={"training_consent": True},
                headers={"Authorization": f"Bearer {token}"},
            )

        assert response.status_code == 200
        assert response.json()["training_consent"] is True
        repo.set_training_consent.assert_called_once_with("existing-user-id", True)
