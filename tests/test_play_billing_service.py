"""Google Play 영수증 검증 서비스 테스트 (Google API 모킹)"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")

from unittest.mock import MagicMock, patch

import pytest

from services.play_billing_service import (
    InvalidPurchaseError,
    PlayBillingService,
    PlayBillingUnavailableError,
)


def _response(status_code: int, body: dict = None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body or {}
    return resp


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv("PLAY_PACKAGE_NAME", "com.hairme.app")
    svc = PlayBillingService()
    svc._session = MagicMock()
    return svc


class TestVerifyProductPurchase:
    def test_valid_purchase(self, service):
        service._session.get.return_value = _response(
            200,
            {
                "purchaseState": 0,
                "orderId": "GPA.1234-5678",
                "purchaseTimeMillis": "1720000000000",
            },
        )

        receipt = service.verify_product_purchase("credits_10", "token-abc")

        assert receipt["order_id"] == "GPA.1234-5678"
        # 요청 URL에 패키지명/상품ID/토큰이 포함되는지 확인
        url = service._session.get.call_args.args[0]
        assert "com.hairme.app" in url
        assert "credits_10" in url
        assert "token-abc" in url

    def test_canceled_purchase_rejected(self, service):
        service._session.get.return_value = _response(200, {"purchaseState": 1})

        with pytest.raises(InvalidPurchaseError):
            service.verify_product_purchase("credits_10", "token-abc")

    def test_pending_purchase_rejected(self, service):
        service._session.get.return_value = _response(200, {"purchaseState": 2})

        with pytest.raises(InvalidPurchaseError):
            service.verify_product_purchase("credits_10", "token-abc")

    def test_unknown_token_rejected(self, service):
        """존재하지 않는 토큰은 Google이 404를 반환"""
        service._session.get.return_value = _response(404)

        with pytest.raises(InvalidPurchaseError):
            service.verify_product_purchase("credits_10", "bad-token")

    def test_google_api_error_maps_to_unavailable(self, service):
        service._session.get.return_value = _response(500)

        with pytest.raises(PlayBillingUnavailableError):
            service.verify_product_purchase("credits_10", "token-abc")

    def test_network_error_maps_to_unavailable(self, service):
        service._session.get.side_effect = ConnectionError("timeout")

        with pytest.raises(PlayBillingUnavailableError):
            service.verify_product_purchase("credits_10", "token-abc")

    def test_missing_package_name(self, monkeypatch):
        monkeypatch.delenv("PLAY_PACKAGE_NAME", raising=False)
        svc = PlayBillingService()
        svc._session = MagicMock()

        with patch("services.play_billing_service.settings") as mock_settings:
            mock_settings.PLAY_PACKAGE_NAME = ""
            with pytest.raises(PlayBillingUnavailableError):
                svc.verify_product_purchase("credits_10", "token-abc")


class TestCredentialLoading:
    def test_missing_service_account_key(self, monkeypatch):
        monkeypatch.delenv("PLAY_SERVICE_ACCOUNT_JSON", raising=False)
        svc = PlayBillingService()

        with patch(
            "services.play_billing_service.get_secret_or_env", return_value=None
        ), patch("services.play_billing_service.settings") as mock_settings:
            mock_settings.PLAY_SERVICE_ACCOUNT_JSON = ""
            mock_settings.AWS_REGION = "ap-northeast-2"
            with pytest.raises(PlayBillingUnavailableError):
                svc._load_credentials()

    def test_malformed_service_account_key(self):
        svc = PlayBillingService()

        with patch(
            "services.play_billing_service.get_secret_or_env",
            return_value="not-a-json",
        ):
            with pytest.raises(PlayBillingUnavailableError):
                svc._load_credentials()
