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
                "acknowledgementState": 0,
            },
        )

        receipt = service.verify_product_purchase("credits_10", "token-abc")

        assert receipt["order_id"] == "GPA.1234-5678"
        assert receipt["acknowledgement_state"] == 0
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


class TestAcknowledgeProductPurchase:
    """H1: 서버 측 acknowledge (미승인 구매의 3일 후 자동 환불 차단)"""

    def test_acknowledge_success(self, service):
        service._session.post.return_value = _response(200)

        service.acknowledge_product_purchase("credits_10", "token-abc")

        url = service._session.post.call_args.args[0]
        assert url.endswith(":acknowledge")
        assert "credits_10" in url
        assert "token-abc" in url

    def test_already_acknowledged_400_is_not_fatal(self, service):
        """이미 승인된 구매(동시 요청/앱 선승인)의 400은 성공으로 간주"""
        service._session.post.return_value = _response(400)

        service.acknowledge_product_purchase("credits_10", "token-abc")  # no raise

    def test_server_error_raises_unavailable(self, service):
        service._session.post.return_value = _response(500)

        with pytest.raises(PlayBillingUnavailableError):
            service.acknowledge_product_purchase("credits_10", "token-abc")

    def test_network_error_raises_unavailable(self, service):
        service._session.post.side_effect = ConnectionError("boom")

        with pytest.raises(PlayBillingUnavailableError):
            service.acknowledge_product_purchase("credits_10", "token-abc")


class TestListVoidedPurchases:
    """환불/취소 구매 목록 조회 (purchases.voidedpurchases.list)"""

    def test_request_params(self, service):
        service._session.get.return_value = _response(
            200,
            {
                "voidedPurchases": [{"purchaseToken": "tok-1"}],
                "tokenPagination": {"nextPageToken": "page-2"},
            },
        )

        data = service.list_voided_purchases(1720000000000)

        assert data["voidedPurchases"][0]["purchaseToken"] == "tok-1"
        url = service._session.get.call_args.args[0]
        params = service._session.get.call_args.kwargs["params"]
        assert "com.hairme.app" in url and url.endswith("voidedpurchases")
        assert params["startTime"] == "1720000000000"
        assert params["type"] == 0  # 인앱 상품(일회성)만
        assert "token" not in params  # 첫 페이지

    def test_page_token_is_sent(self, service):
        service._session.get.return_value = _response(200, {})

        service.list_voided_purchases(1720000000000, page_token="page-2")

        assert service._session.get.call_args.kwargs["params"]["token"] == "page-2"

    @pytest.mark.parametrize("status", [401, 403, 500])
    def test_api_error_raises_unavailable(self, service, status):
        service._session.get.return_value = _response(status)

        with pytest.raises(PlayBillingUnavailableError):
            service.list_voided_purchases(1720000000000)

    def test_network_error_raises_unavailable(self, service):
        service._session.get.side_effect = ConnectionError("boom")

        with pytest.raises(PlayBillingUnavailableError):
            service.list_voided_purchases(1720000000000)

    def test_missing_package_name_raises_unavailable(self, monkeypatch):
        monkeypatch.setenv("PLAY_PACKAGE_NAME", "")
        monkeypatch.setattr(
            "services.play_billing_service.settings.PLAY_PACKAGE_NAME", ""
        )
        svc = PlayBillingService()
        svc._session = MagicMock()

        with pytest.raises(PlayBillingUnavailableError):
            svc.list_voided_purchases(1720000000000)
