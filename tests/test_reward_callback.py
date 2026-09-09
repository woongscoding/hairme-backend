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

from config.settings import settings
from main import app
from services.admob_ssv_service import (
    AdMobSSVService,
    InvalidSSVError,
    REWARD_MAX_AGE_SECONDS,
    SSVUnavailableError,
    is_reward_timestamp_fresh,
    parse_ad_unit_allowlist,
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

    def test_appended_params_after_signature_rejected(self):
        """서명 뒤에 &user_id=... 를 덧붙여 서명된 값을 덮어쓰는 우회 차단"""
        svc = _service_with_cached_key()
        raw = _signed_query(SSV_MESSAGE) + b"&user_id=attacker&transaction_id=tx-evil"

        with pytest.raises(InvalidSSVError):
            svc.verify_callback(raw)

    def test_signed_values_win_over_appended_params(self):
        """혹시 통과하더라도 반환값은 서명 구간의 값이어야 한다"""
        svc = _service_with_cached_key()
        raw = _signed_query(SSV_MESSAGE)

        params = svc.verify_callback(raw)

        assert params["user_id"] == "reward-user-id"
        assert "signature" not in params
        assert "key_id" not in params

    def test_appended_param_before_key_id_rejected(self):
        """signature와 key_id 사이에 파라미터를 끼워 넣는 경우도 차단"""
        svc = _service_with_cached_key()
        raw = _signed_query(SSV_MESSAGE).replace(
            b"&key_id=", b"&user_id=attacker&key_id="
        )

        with pytest.raises(InvalidSSVError):
            svc.verify_callback(raw)

    def test_duplicate_key_in_signed_portion_rejected(self):
        """서명 구간에 중복 키가 있으면 어떤 값이 서명됐는지 모호 → 거부"""
        svc = _service_with_cached_key()
        message = SSV_MESSAGE + "&user_id=attacker"

        with pytest.raises(InvalidSSVError):
            svc.verify_callback(_signed_query(message))

    def test_duplicate_key_id_after_signature_rejected(self):
        svc = _service_with_cached_key()
        raw = _signed_query(SSV_MESSAGE) + f"&key_id={TEST_KEY_ID}".encode("utf-8")

        with pytest.raises(InvalidSSVError):
            svc.verify_callback(raw)

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

    def test_grant_failure_restores_claim_and_counter(
        self, client, mock_ssv, mock_credit, mock_usage
    ):
        """L4: 지급 실패 시 클레임 회수 + 일일 카운터 복구
        (AdMob 재시도가 사용자 보상 한도를 소모하지 않도록)"""
        mock_credit.grant.side_effect = Exception("DynamoDB down")

        response = client.get(CALLBACK_URL)

        assert response.status_code == 500
        mock_credit.release_ref.assert_called_once_with("reward#tx-abc-123")
        mock_usage.decrement_daily_counter.assert_called_once_with(
            "reward_ad#reward-user-id"
        )
        assert "DynamoDB" not in response.text

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


class TestRewardFreshnessHelpers:
    def test_fresh_timestamp(self):
        now = 1_720_000_000.0
        assert is_reward_timestamp_fresh(str(int(now * 1000)), now_seconds=now) is True

    def test_stale_timestamp(self):
        now = 1_720_000_000.0
        stale_ms = int((now - REWARD_MAX_AGE_SECONDS - 60) * 1000)
        assert is_reward_timestamp_fresh(str(stale_ms), now_seconds=now) is False

    def test_far_future_timestamp(self):
        now = 1_720_000_000.0
        future_ms = int((now + 3600) * 1000)
        assert is_reward_timestamp_fresh(str(future_ms), now_seconds=now) is False

    def test_unparsable_timestamp(self):
        assert is_reward_timestamp_fresh("not-a-number") is False

    def test_allowlist_parsing(self):
        assert parse_ad_unit_allowlist(" a/1 , b/2 ,, ") == {"a/1", "b/2"}
        assert parse_ad_unit_allowlist("") == set()
        assert parse_ad_unit_allowlist(None) == set()


class TestAdUnitAllowlist:
    def test_ad_unit_not_in_allowlist_rejected(
        self, client, mock_ssv, mock_credit, mock_usage, monkeypatch
    ):
        monkeypatch.setattr(settings, "ADMOB_REWARD_AD_UNIT_IDS", "ca-app-pub-1/999")

        response = client.get(CALLBACK_URL)

        assert response.status_code == 400
        mock_credit.try_claim_ref.assert_not_called()
        mock_credit.grant.assert_not_called()

    def test_ad_unit_in_allowlist_allowed(
        self, client, mock_ssv, mock_credit, mock_usage, monkeypatch
    ):
        monkeypatch.setattr(
            settings, "ADMOB_REWARD_AD_UNIT_IDS", " other/1 , 1234567890 "
        )

        response = client.get(CALLBACK_URL)

        assert response.status_code == 200
        assert response.json()["rewarded"] is True

    def test_empty_allowlist_fails_closed_in_production(
        self, client, mock_ssv, mock_credit, mock_usage, monkeypatch
    ):
        monkeypatch.setattr(settings, "ADMOB_REWARD_AD_UNIT_IDS", "")
        monkeypatch.setattr(settings, "ENVIRONMENT", "production")

        response = client.get(CALLBACK_URL)

        assert response.status_code == 503
        mock_credit.grant.assert_not_called()

    def test_empty_allowlist_allowed_in_development(
        self, client, mock_ssv, mock_credit, mock_usage, monkeypatch
    ):
        monkeypatch.setattr(settings, "ADMOB_REWARD_AD_UNIT_IDS", "")
        monkeypatch.setattr(settings, "ENVIRONMENT", "development")

        response = client.get(CALLBACK_URL)

        assert response.status_code == 200


class TestCallbackFreshness:
    def test_stale_timestamp_rejected(
        self, client, mock_ssv, mock_credit, mock_usage, monkeypatch
    ):
        monkeypatch.setattr(settings, "ADMOB_REWARD_AD_UNIT_IDS", "1234567890")
        stale_ms = int((time.time() - REWARD_MAX_AGE_SECONDS - 120) * 1000)
        mock_ssv.verify_callback.return_value = {
            "user_id": "reward-user-id",
            "transaction_id": "tx-abc-123",
            "ad_unit": "1234567890",
            "timestamp": str(stale_ms),
        }

        response = client.get(CALLBACK_URL)

        assert response.status_code == 400
        mock_credit.grant.assert_not_called()

    def test_fresh_timestamp_accepted(
        self, client, mock_ssv, mock_credit, mock_usage, monkeypatch
    ):
        monkeypatch.setattr(settings, "ADMOB_REWARD_AD_UNIT_IDS", "1234567890")
        mock_ssv.verify_callback.return_value = {
            "user_id": "reward-user-id",
            "transaction_id": "tx-abc-123",
            "ad_unit": "1234567890",
            "timestamp": str(int(time.time() * 1000)),
        }

        response = client.get(CALLBACK_URL)

        assert response.status_code == 200
        assert response.json()["rewarded"] is True

    def test_missing_timestamp_allowed(self, client, mock_ssv, mock_credit, mock_usage):
        response = client.get(CALLBACK_URL)

        assert response.status_code == 200
