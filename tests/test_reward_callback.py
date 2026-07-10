"""AdMob 리워드 광고 SSV 콜백 테스트 (서명은 실제 ECDSA 키로 생성)"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

import base64
import time
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from main import app
from services.admob_ssv_service import (
    AdMobSSVService,
    InvalidSSVError,
    SSVUnavailableError,
)
from services.usage_limit_service import UsageLimitService

TEST_KEY_ID = "3335741209"

# 테스트용 ECDSA P-256 키쌍 (모듈 로드 시 1회 생성)
_PRIVATE_KEY = ec.generate_private_key(ec.SECP256R1())
_PUBLIC_PEM = (
    _PRIVATE_KEY.public_key()
    .public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    .decode("utf-8")
)


def _signed_query(message: str, key_id: str = TEST_KEY_ID) -> bytes:
    """AdMob이 보내는 형식의 서명된 쿼리 스트링 생성"""
    signature = _PRIVATE_KEY.sign(message.encode("utf-8"), ec.ECDSA(hashes.SHA256()))
    sig_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("utf-8")
    return f"{message}&signature={sig_b64}&key_id={key_id}".encode("utf-8")


def _service_with_cached_key() -> AdMobSSVService:
    svc = AdMobSSVService()
    svc._keys = {TEST_KEY_ID: _PUBLIC_PEM}
    svc._keys_fetched_at = time.monotonic()
    return svc


SSV_MESSAGE = (
    "ad_network=5450213213286189855&ad_unit=1234567890&reward_amount=1"
    "&reward_item=credit&timestamp=1720000000000"
    "&transaction_id=tx-abc-123&user_id=reward-user-id"
)


class TestAdMobSSVService:
    def test_valid_signature(self):
        svc = _service_with_cached_key()

        params = svc.verify_callback(_signed_query(SSV_MESSAGE))

        assert params["user_id"] == "reward-user-id"
        assert params["transaction_id"] == "tx-abc-123"

    def test_tampered_message_rejected(self):
        """파라미터 변조 (보상 횟수 부풀리기 등) 시 서명 불일치"""
        svc = _service_with_cached_key()
        raw = _signed_query(SSV_MESSAGE).replace(b"reward_amount=1", b"reward_amount=9")

        with pytest.raises(InvalidSSVError):
            svc.verify_callback(raw)

    def test_missing_signature_rejected(self):
        svc = _service_with_cached_key()

        with pytest.raises(InvalidSSVError):
            svc.verify_callback(SSV_MESSAGE.encode("utf-8"))

    def test_unknown_key_id_refetches_then_rejects(self):
        """캐시에 없는 key_id는 키 회전 대비 재조회 후, 그래도 없으면 거부"""
        svc = _service_with_cached_key()
        svc._fetch_keys = MagicMock(return_value={TEST_KEY_ID: _PUBLIC_PEM})

        with pytest.raises(InvalidSSVError):
            svc.verify_callback(_signed_query(SSV_MESSAGE, key_id="9999"))

        svc._fetch_keys.assert_called_once()

    def test_keys_fetch_failure_maps_to_unavailable(self):
        svc = AdMobSSVService()  # 캐시 비어 있음

        with patch("services.admob_ssv_service.httpx.get", side_effect=ConnectionError):
            with pytest.raises(SSVUnavailableError):
                svc.verify_callback(_signed_query(SSV_MESSAGE))

    def test_keys_parsed_from_google_response(self):
        svc = AdMobSSVService()
        response = MagicMock()
        response.json.return_value = {
            "keys": [{"keyId": int(TEST_KEY_ID), "pem": _PUBLIC_PEM, "base64": "..."}]
        }

        with patch("services.admob_ssv_service.httpx.get", return_value=response):
            params = svc.verify_callback(_signed_query(SSV_MESSAGE))

        assert params["transaction_id"] == "tx-abc-123"


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_ssv():
    """서명 검증 성공을 기본값으로 모킹"""
    service = MagicMock()
    service.verify_callback.return_value = {
        "user_id": "reward-user-id",
        "transaction_id": "tx-abc-123",
        "ad_unit": "1234567890",
    }
    with patch("api.endpoints.credits.get_admob_ssv_service", return_value=service):
        yield service


@pytest.fixture
def mock_credit():
    service = MagicMock()
    service.try_claim_ref.return_value = True
    service.grant.return_value = 6
    with patch("api.endpoints.credits.get_credit_service", return_value=service):
        yield service


@pytest.fixture
def mock_usage():
    """일일 상한 미달을 기본값으로 모킹"""
    service = MagicMock()
    service.increment_daily_counter.return_value = True
    with patch("api.endpoints.credits.get_usage_limit_service", return_value=service):
        yield service


CALLBACK_URL = "/api/credits/reward-callback?user_id=reward-user-id&transaction_id=tx-abc-123&signature=sig&key_id=123"


class TestRewardCallback:
    def test_successful_reward(self, client, mock_ssv, mock_credit, mock_usage):
        response = client.get(CALLBACK_URL)

        assert response.status_code == 200
        assert response.json()["rewarded"] is True
        # +1 크레딧, reason="reward_ad", transaction_id로 추적
        mock_credit.grant.assert_called_once_with(
            "reward-user-id", 1, reason="reward_ad", ref_id="tx-abc-123"
        )
        # 일일 상한 카운터는 유저 네임스페이스 키로 증가
        counter_args = mock_usage.increment_daily_counter.call_args.args
        assert counter_args[0] == "reward_ad#reward-user-id"

    def test_invalid_signature(self, client, mock_ssv, mock_credit, mock_usage):
        mock_ssv.verify_callback.side_effect = InvalidSSVError()

        response = client.get(CALLBACK_URL)

        assert response.status_code == 400
        mock_credit.grant.assert_not_called()

    def test_verifier_keys_unavailable(self, client, mock_ssv, mock_credit, mock_usage):
        mock_ssv.verify_callback.side_effect = SSVUnavailableError()

        response = client.get(CALLBACK_URL)

        assert response.status_code == 503
        mock_credit.grant.assert_not_called()

    def test_missing_user_id(self, client, mock_ssv, mock_credit, mock_usage):
        """앱이 SSV 옵션에 user_id를 설정하지 않으면 지급 불가"""
        mock_ssv.verify_callback.return_value = {"transaction_id": "tx-abc-123"}

        response = client.get(CALLBACK_URL)

        assert response.status_code == 400
        mock_credit.grant.assert_not_called()

    def test_duplicate_transaction(self, client, mock_ssv, mock_credit, mock_usage):
        """같은 transaction_id 재전송 시 중복 지급 없이 200 (AdMob 재시도 중단)"""
        mock_credit.try_claim_ref.return_value = False

        response = client.get(CALLBACK_URL)

        assert response.status_code == 200
        assert response.json()["rewarded"] is False
        mock_credit.grant.assert_not_called()
        mock_usage.increment_daily_counter.assert_not_called()

    def test_daily_limit_reached(self, client, mock_ssv, mock_credit, mock_usage):
        mock_usage.increment_daily_counter.return_value = False

        response = client.get(CALLBACK_URL)

        assert response.status_code == 200
        data = response.json()
        assert data["rewarded"] is False
        assert data["reason"] == "daily_limit_reached"
        mock_credit.grant.assert_not_called()

    def test_grant_failure_releases_claim(
        self, client, mock_ssv, mock_credit, mock_usage
    ):
        """지급 실패 시 클레임 회수 후 500 → AdMob 재시도에서 재처리"""
        mock_credit.grant.side_effect = Exception("DynamoDB down")

        response = client.get(CALLBACK_URL)

        assert response.status_code == 500
        mock_credit.release_ref.assert_called_once_with("reward#tx-abc-123")
        assert "DynamoDB" not in response.text

    def test_unknown_user_returns_200(self, client, mock_ssv, mock_credit, mock_usage):
        """탈퇴한 사용자 등 - 재시도해도 소용없으므로 200으로 재시도 중단"""
        mock_credit.grant.side_effect = ValueError("존재하지 않는 사용자입니다")

        response = client.get(CALLBACK_URL)

        assert response.status_code == 200
        assert response.json()["rewarded"] is False
        mock_credit.release_ref.assert_called_once()


class TestIncrementDailyCounter:
    @pytest.fixture
    def service(self):
        svc = UsageLimitService()
        svc._table = MagicMock()
        return svc

    def test_under_limit_increments(self, service):
        result = service.increment_daily_counter("reward_ad#user-1", 5)

        assert result is True
        call_kwargs = service._table.update_item.call_args.kwargs
        # 조건부 원자 증가인지 확인
        assert "#cnt < :limit" in call_kwargs["ConditionExpression"]
        assert call_kwargs["ExpressionAttributeValues"][":limit"] == 5
        assert call_kwargs["Key"]["device_id"] == "reward_ad#user-1"

    def test_limit_reached_returns_false(self, service):
        service._table.update_item.side_effect = ClientError(
            error_response={
                "Error": {
                    "Code": "ConditionalCheckFailedException",
                    "Message": "The conditional request failed",
                }
            },
            operation_name="UpdateItem",
        )

        assert service.increment_daily_counter("reward_ad#user-1", 5) is False
