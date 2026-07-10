"""크레딧 구매 엔드포인트 테스트 (Play API + DynamoDB 모킹)"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from core.jwt_auth import create_access_token
from main import app
from services.play_billing_service import (
    InvalidPurchaseError,
    PlayBillingUnavailableError,
)

PURCHASE_BODY = {"product_id": "credits_10", "purchase_token": "token-abc-123"}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers():
    token = create_access_token("buyer-user-id")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def mock_play():
    """영수증 검증 성공을 기본값으로 모킹"""
    service = MagicMock()
    service.verify_product_purchase.return_value = {
        "order_id": "GPA.1234-5678",
        "purchase_time_millis": "1720000000000",
    }
    with patch("api.endpoints.credits.get_play_billing_service", return_value=service):
        yield service


@pytest.fixture
def mock_credit():
    """클레임 성공 + 지급 성공을 기본값으로 모킹"""
    service = MagicMock()
    service.try_claim_ref.return_value = True
    service.grant.return_value = 15
    with patch("api.endpoints.credits.get_credit_service", return_value=service):
        yield service


class TestPurchaseCredits:
    def test_successful_purchase(self, client, auth_headers, mock_play, mock_credit):
        response = client.post(
            "/api/credits/purchase", json=PURCHASE_BODY, headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["granted"] == 10
        assert data["balance"] == 15
        assert data["order_id"] == "GPA.1234-5678"

        # reason="purchase", ref_id=purchase_token으로 지급되는지 확인
        grant_kwargs = mock_credit.grant.call_args.kwargs
        assert grant_kwargs["reason"] == "purchase"
        assert grant_kwargs["ref_id"] == "token-abc-123"
        assert mock_credit.grant.call_args.args == ("buyer-user-id", 10)

    def test_requires_auth(self, client, mock_play, mock_credit):
        response = client.post("/api/credits/purchase", json=PURCHASE_BODY)
        assert response.status_code == 401

    def test_unknown_product(self, client, auth_headers, mock_play, mock_credit):
        response = client.post(
            "/api/credits/purchase",
            json={"product_id": "credits_9999", "purchase_token": "token"},
            headers=auth_headers,
        )
        assert response.status_code == 400
        mock_play.verify_product_purchase.assert_not_called()
        mock_credit.grant.assert_not_called()

    def test_invalid_receipt(self, client, auth_headers, mock_play, mock_credit):
        mock_play.verify_product_purchase.side_effect = InvalidPurchaseError()

        response = client.post(
            "/api/credits/purchase", json=PURCHASE_BODY, headers=auth_headers
        )

        assert response.status_code == 400
        mock_credit.grant.assert_not_called()

    def test_play_api_unavailable(self, client, auth_headers, mock_play, mock_credit):
        mock_play.verify_product_purchase.side_effect = PlayBillingUnavailableError()

        response = client.post(
            "/api/credits/purchase", json=PURCHASE_BODY, headers=auth_headers
        )

        assert response.status_code == 503
        mock_credit.grant.assert_not_called()

    def test_duplicate_token_rejected(
        self, client, auth_headers, mock_play, mock_credit
    ):
        """같은 purchase_token 재사용 시 중복 지급 방지 (멱등성)"""
        mock_credit.try_claim_ref.return_value = False

        response = client.post(
            "/api/credits/purchase", json=PURCHASE_BODY, headers=auth_headers
        )

        assert response.status_code == 409
        mock_credit.grant.assert_not_called()

    def test_grant_failure_releases_claim(
        self, client, auth_headers, mock_play, mock_credit
    ):
        """지급 실패 시 클레임 마커를 회수해 재시도 가능하게 함"""
        mock_credit.grant.side_effect = Exception("DynamoDB down")

        response = client.post(
            "/api/credits/purchase", json=PURCHASE_BODY, headers=auth_headers
        )

        assert response.status_code == 500
        mock_credit.release_ref.assert_called_once_with("purchase#token-abc-123")
        # 내부 오류 정보가 응답에 노출되지 않음
        assert "DynamoDB" not in response.text

    def test_missing_fields(self, client, auth_headers):
        response = client.post("/api/credits/purchase", json={}, headers=auth_headers)
        assert response.status_code == 422
