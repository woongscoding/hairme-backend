"""회원 인증 엔드포인트 테스트 (카카오 API + DynamoDB 모킹)"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import jwt as pyjwt
import pytest
from fastapi.testclient import TestClient

from config.settings import settings
from core.jwt_auth import (
    TOKEN_TYPE_REFRESH,
    _create_token,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from database.user_repository import UserAlreadyExistsError
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
    "token_version": 1,
    "created_at": "2026-07-01T00:00:00+00:00",
}


def _repo_with_user(user: dict) -> MagicMock:
    """get_by_id / get_by_id_consistent 가 같은 사용자를 돌려주는 리포지토리 목"""
    repo = MagicMock()
    repo.get_by_id.return_value = dict(user)
    repo.get_by_id_consistent.return_value = dict(user)
    return repo


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

    def test_signup_race_falls_back_to_existing_user_without_bonus(
        self, client, mock_kakao
    ):
        """동시 가입 경합: create가 거부되면 기존 계정 로그인으로 전환(보너스 없음)"""
        repo = MagicMock()
        # 첫 조회는 GSI 지연으로 None, create는 유일성 마커 조건 실패
        repo.get_by_kakao_id.side_effect = [None, dict(EXISTING_USER)]
        repo.create.side_effect = UserAlreadyExistsError("12345678")

        credit_service = MagicMock()

        with patch("api.endpoints.auth.get_user_repository", return_value=repo), patch(
            "api.endpoints.auth.get_credit_service", return_value=credit_service
        ):
            response = client.post(
                "/api/auth/kakao", json={"kakao_access_token": "valid_kakao_token"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["is_new_user"] is False
        assert data["user"]["user_id"] == "existing-user-id"
        assert data["user"]["credits"] == 3
        # 가입 보너스는 지급되지 않아야 한다 (중복 지급 방지)
        credit_service.grant.assert_not_called()
        repo.update_last_login.assert_called_once_with("existing-user-id")

    def test_signup_race_falls_back_to_uniqueness_marker(self, client, mock_kakao):
        """GSI가 계속 비어 있으면 강한 일관성 마커 조회로 기존 계정을 찾는다"""
        repo = MagicMock()
        repo.get_by_kakao_id.return_value = None
        repo.create.side_effect = UserAlreadyExistsError("12345678")
        repo.get_by_kakao_marker.return_value = dict(EXISTING_USER)

        credit_service = MagicMock()

        with patch("api.endpoints.auth.get_user_repository", return_value=repo), patch(
            "api.endpoints.auth.get_credit_service", return_value=credit_service
        ):
            response = client.post(
                "/api/auth/kakao", json={"kakao_access_token": "valid_kakao_token"}
            )

        assert response.status_code == 200
        assert response.json()["is_new_user"] is False
        repo.get_by_kakao_marker.assert_called_once_with("12345678")
        credit_service.grant.assert_not_called()

    def test_signup_race_unresolvable_returns_503(self, client, mock_kakao):
        repo = MagicMock()
        repo.get_by_kakao_id.return_value = None
        repo.create.side_effect = UserAlreadyExistsError("12345678")
        repo.get_by_kakao_marker.return_value = None

        credit_service = MagicMock()

        with patch("api.endpoints.auth.get_user_repository", return_value=repo), patch(
            "api.endpoints.auth.get_credit_service", return_value=credit_service
        ):
            response = client.post(
                "/api/auth/kakao", json={"kakao_access_token": "valid_kakao_token"}
            )

        assert response.status_code == 503
        credit_service.grant.assert_not_called()

    def test_login_suspended_user_rejected(self, client, mock_kakao):
        """정지 계정은 토큰을 발급받지 못한다 (403)"""
        repo = MagicMock()
        repo.get_by_kakao_id.return_value = dict(EXISTING_USER, status="suspended")

        with patch("api.endpoints.auth.get_user_repository", return_value=repo):
            response = client.post(
                "/api/auth/kakao", json={"kakao_access_token": "valid_kakao_token"}
            )

        assert response.status_code == 403
        assert "정지" in response.json()["detail"]
        repo.update_last_login.assert_not_called()

    def test_login_issues_tokens_with_user_token_version(self, client, mock_kakao):
        repo = MagicMock()
        repo.get_by_kakao_id.return_value = dict(EXISTING_USER, token_version=4)

        with patch("api.endpoints.auth.get_user_repository", return_value=repo):
            response = client.post(
                "/api/auth/kakao", json={"kakao_access_token": "valid_kakao_token"}
            )

        assert response.status_code == 200
        data = response.json()
        assert decode_token(data["access_token"])["tv"] == 4
        assert decode_token(data["refresh_token"], TOKEN_TYPE_REFRESH)["tv"] == 4

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
    def test_refresh_returns_new_token_pair(self, client):
        repo = _repo_with_user(dict(EXISTING_USER, user_id="user-123"))
        refresh = create_refresh_token("user-123", 1)

        with patch("api.endpoints.auth.get_user_repository", return_value=repo):
            response = client.post("/api/auth/refresh", json={"refresh_token": refresh})

        assert response.status_code == 200
        data = response.json()
        assert data["access_token"]
        assert data["refresh_token"]
        # 강한 일관성 조회로 token_version/status 확인
        repo.get_by_id_consistent.assert_called_once_with("user-123")

    def test_access_token_rejected_for_refresh(self, client):
        """액세스 토큰으로는 재발급 불가"""
        access = create_access_token("user-123")
        response = client.post("/api/auth/refresh", json={"refresh_token": access})
        assert response.status_code == 401

    def test_refresh_rejected_when_token_version_mismatch(self, client):
        """logout-all 등으로 token_version 이 올라간 뒤의 옛 리프레시 토큰 → 401"""
        repo = _repo_with_user(dict(EXISTING_USER, user_id="user-123", token_version=2))
        stale = create_refresh_token("user-123", 1)

        with patch("api.endpoints.auth.get_user_repository", return_value=repo):
            response = client.post("/api/auth/refresh", json={"refresh_token": stale})

        assert response.status_code == 401
        assert "다시 로그인" in response.json()["detail"]

    def test_refresh_rejected_for_suspended_user(self, client):
        repo = _repo_with_user(dict(EXISTING_USER, status="suspended"))
        refresh = create_refresh_token(EXISTING_USER["user_id"], 1)

        with patch("api.endpoints.auth.get_user_repository", return_value=repo):
            response = client.post("/api/auth/refresh", json={"refresh_token": refresh})

        assert response.status_code == 401
        assert "정지" in response.json()["detail"]

    def test_refresh_rejected_when_user_missing(self, client):
        repo = MagicMock()
        repo.get_by_id_consistent.return_value = None
        refresh = create_refresh_token("deleted-user", 1)

        with patch("api.endpoints.auth.get_user_repository", return_value=repo):
            response = client.post("/api/auth/refresh", json={"refresh_token": refresh})

        assert response.status_code == 401

    def test_legacy_token_without_tv_still_refreshes(self, client):
        """tv 클레임이 없는 과거 토큰은 tv=1 로 간주 (기존 세션 유지)"""
        issued = _create_token(
            EXISTING_USER["user_id"], TOKEN_TYPE_REFRESH, timedelta(days=30)
        )
        claims = pyjwt.decode(
            issued, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        claims.pop("tv")
        legacy = pyjwt.encode(
            claims, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        )

        # token_version 속성이 아예 없는 기존 회원도 1로 취급된다
        user = dict(EXISTING_USER)
        user.pop("token_version")
        repo = _repo_with_user(user)

        with patch("api.endpoints.auth.get_user_repository", return_value=repo):
            response = client.post("/api/auth/refresh", json={"refresh_token": legacy})

        assert response.status_code == 200
        assert response.json()["access_token"]


class TestLogoutAll:
    def test_logout_all_requires_auth(self, client):
        assert client.post("/api/auth/logout-all").status_code == 401

    def test_logout_all_bumps_token_version_and_kills_old_refresh(self, client):
        repo = _repo_with_user(EXISTING_USER)
        repo.bump_token_version.return_value = 2
        access = create_access_token(EXISTING_USER["user_id"], 1)
        old_refresh = create_refresh_token(EXISTING_USER["user_id"], 1)

        with patch("api.endpoints.auth.get_user_repository", return_value=repo):
            response = client.post(
                "/api/auth/logout-all",
                headers={"Authorization": f"Bearer {access}"},
            )

            assert response.status_code == 200
            assert response.json()["token_version"] == 2
            repo.bump_token_version.assert_called_once_with(EXISTING_USER["user_id"])

            # 증가된 버전을 반영한 뒤 옛 리프레시 토큰을 시도하면 거부된다
            repo.get_by_id_consistent.return_value = dict(
                EXISTING_USER, token_version=2
            )
            refresh_response = client.post(
                "/api/auth/refresh", json={"refresh_token": old_refresh}
            )

        assert refresh_response.status_code == 401


class TestAdminUserStatus:
    HEADERS = {"X-API-Key": "test-admin-key"}

    @pytest.fixture(autouse=True)
    def admin_key(self, monkeypatch):
        monkeypatch.setattr(settings, "ADMIN_API_KEY", "test-admin-key")

    def test_suspend_requires_admin_key(self, client):
        response = client.post("/api/admin/users/u1/suspend")
        assert response.status_code == 403

    def test_suspend_sets_status_and_bumps_version(self, client):
        repo = MagicMock()
        repo.bump_token_version.return_value = 3

        with patch("database.user_repository.get_user_repository", return_value=repo):
            response = client.post("/api/admin/users/u1/suspend", headers=self.HEADERS)

        assert response.status_code == 200
        assert response.json()["status"] == "suspended"
        assert response.json()["token_version"] == 3
        repo.set_status.assert_called_once_with("u1", "suspended")
        repo.bump_token_version.assert_called_once_with("u1")

    def test_suspend_unknown_user_returns_404(self, client):
        repo = MagicMock()
        repo.set_status.side_effect = ValueError("존재하지 않는 사용자입니다")

        with patch("database.user_repository.get_user_repository", return_value=repo):
            response = client.post(
                "/api/admin/users/nope/suspend", headers=self.HEADERS
            )

        assert response.status_code == 404

    def test_reactivate_sets_status_active(self, client):
        repo = MagicMock()

        with patch("database.user_repository.get_user_repository", return_value=repo):
            response = client.post(
                "/api/admin/users/u1/reactivate", headers=self.HEADERS
            )

        assert response.status_code == 200
        assert response.json()["status"] == "active"
        repo.set_status.assert_called_once_with("u1", "active")
        # 정지 해제로 token_version 을 되돌리지는 않는다
        repo.bump_token_version.assert_not_called()


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
