# -*- coding: utf-8 -*-
"""합성 비용 보호 테스트 (작업 3)

- 중복 요청 방지: 같은 주체 + 같은 cache_key 가 진행 중이면 두 번째 요청은 409.
  Lambda 는 인스턴스가 여러 개라 in-process 잠금이 안 되므로 DynamoDB
  조건부 쓰기로 구현하고, 기존 hairstyle_usage 테이블을 재사용한다.
- 잠금 확인은 크레딧 차감보다 앞에 둔다 (거절된 요청은 과금되면 안 된다).
- 잠금 저장소 장애 시에는 합성을 막지 않고 통과시키되 경고 로그를 남긴다.
- 캐시 히트는 과금도 API 호출도 없으므로 잠금에서 제외한다.
- 재시도 비용: 실제 Gemini 호출 횟수를 api_calls 로 구조화 로그에 남긴다.

외부 호출(DynamoDB / S3 / Gemini)은 전부 모킹한다.
"""

import io
import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from core.jwt_auth import create_access_token
from core.synthesis_lock import (
    LOCK_SORT_KEY,
    LOCK_TTL_SECONDS,
    acquire_synthesis_lock,
    build_lock_key,
)
from main import app

DEVICE_ID = "a1b2c3d4e5f60718"
USER_ID = "synthesis-lock-user"
CACHE_KEY = "c" * 64
AUTH = {"Authorization": f"Bearer {create_access_token(USER_ID)}"}


def _client_error(code="ConditionalCheckFailedException"):
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": code, "Message": code}}, "PutItem")


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def _usage_service(table=None):
    service = MagicMock()
    service.table = table if table is not None else MagicMock()
    return service


# ========== (a) 잠금 모듈 ==========


class TestSynthesisLock:
    def test_lock_key_hides_the_subject(self):
        key = build_lock_key(f"user:{USER_ID}", CACHE_KEY)

        assert key.startswith("synlock#")
        assert USER_ID not in key
        # 같은 주체 + 같은 cache_key 는 같은 키
        assert key == build_lock_key(f"user:{USER_ID}", CACHE_KEY)
        # 다른 주체는 다른 키
        assert key != build_lock_key("user:other", CACHE_KEY)
        # 다른 cache_key 도 다른 키
        assert key != build_lock_key(f"user:{USER_ID}", "d" * 64)

    def test_acquire_writes_conditional_item_to_usage_table(self):
        table = MagicMock()
        acquired, release = acquire_synthesis_lock(
            f"user:{USER_ID}",
            CACHE_KEY,
            endpoint="synthesize",
            usage_service_factory=lambda: _usage_service(table),
        )

        assert acquired is True
        kwargs = table.update_item.call_args.kwargs
        values = kwargs["ExpressionAttributeValues"]
        assert kwargs["Key"] == {
            "device_id": build_lock_key(f"user:{USER_ID}", CACHE_KEY),
            "date": LOCK_SORT_KEY,
        }
        # 기존 코드가 이 테이블에 쓰는 연산과 같다 (IAM 변경 불필요)
        table.put_item.assert_not_called()
        # 만료는 TTL 삭제가 아니라 조건식으로 판정한다 (TTL 은 최대 48시간 지연)
        assert kwargs["ExpressionAttributeNames"]["#lock"] == "lock_expires_at"
        assert kwargs["ConditionExpression"] == (
            "attribute_not_exists(#lock) OR #lock < :now"
        )
        assert values[":expires"] - values[":now"] == LOCK_TTL_SECONDS
        # 기존 테이블의 TTL 속성도 채워 버려진 잠금이 청소되도록 한다
        assert values[":item_ttl"] > values[":expires"]

    def test_second_request_is_rejected(self):
        table = MagicMock()
        table.update_item.side_effect = _client_error()

        acquired, _release = acquire_synthesis_lock(
            f"user:{USER_ID}",
            CACHE_KEY,
            usage_service_factory=lambda: _usage_service(table),
        )

        assert acquired is False

    def test_release_deletes_the_item(self):
        table = MagicMock()
        _acquired, release = acquire_synthesis_lock(
            f"user:{USER_ID}",
            CACHE_KEY,
            usage_service_factory=lambda: _usage_service(table),
        )

        release()

        # 아이템을 지우지 않고 만료 시각을 0 으로 내린다 (DeleteItem 권한 불필요)
        release_kwargs = table.update_item.call_args.kwargs
        assert release_kwargs["Key"] == {
            "device_id": build_lock_key(f"user:{USER_ID}", CACHE_KEY),
            "date": LOCK_SORT_KEY,
        }
        assert release_kwargs["ExpressionAttributeValues"] == {":zero": 0}
        table.delete_item.assert_not_called()

    def test_release_failure_is_swallowed(self):
        table = MagicMock()
        table.update_item.side_effect = [None, Exception("DynamoDB down")]
        _acquired, release = acquire_synthesis_lock(
            f"user:{USER_ID}",
            CACHE_KEY,
            usage_service_factory=lambda: _usage_service(table),
        )

        release()  # 예외가 새어 나오면 안 된다

    def test_store_failure_fails_open_with_warning(self):
        table = MagicMock()
        table.update_item.side_effect = _client_error("AccessDeniedException")

        with patch("core.synthesis_lock.log_structured") as mock_log:
            acquired, release = acquire_synthesis_lock(
                f"user:{USER_ID}",
                CACHE_KEY,
                endpoint="synthesize",
                usage_service_factory=lambda: _usage_service(table),
            )

        assert acquired is True  # 합성을 막지 않는다
        release()  # no-op - 잡지도 않은 잠금을 풀려고 하지 않는다
        assert table.update_item.call_count == 1
        assert mock_log.call_args.args[0] == "synthesis_lock_unavailable"

    def test_table_unavailable_fails_open(self):
        """테이블 접근 자체가 실패해도 통과시킨다"""
        service = MagicMock()
        type(service).table = property(
            lambda _self: (_ for _ in ()).throw(RuntimeError("no boto3"))
        )

        with patch("core.synthesis_lock.log_structured"):
            acquired, _release = acquire_synthesis_lock(
                f"user:{USER_ID}", CACHE_KEY, usage_service_factory=lambda: service
            )

        assert acquired is True

    def test_subject_separates_member_from_anonymous_device(self):
        """회원과 비로그인 기기는 서로의 잠금에 걸리지 않는다"""
        from api.endpoints.synthesis import _lock_subject

        assert _lock_subject(USER_ID, None) == f"user:{USER_ID}"
        assert _lock_subject(None, DEVICE_ID) == f"device:{DEVICE_ID}"
        # 로그인 상태면 device_id 가 함께 와도 user_id 를 쓴다
        assert _lock_subject(USER_ID, DEVICE_ID) == f"user:{USER_ID}"
        assert _lock_subject(None, "   ") is None
        assert _lock_subject(None, None) is None

        assert build_lock_key(f"user:{USER_ID}", CACHE_KEY) != build_lock_key(
            f"device:{DEVICE_ID}", CACHE_KEY
        )

    def test_device_id_with_namespace_characters_cannot_forge_a_key(self):
        """주체는 해시로 들어가므로 '#' 을 넣어도 다른 카운터를 건드릴 수 없다"""
        key = build_lock_key("device:ip#203.0.113.42", CACHE_KEY)

        assert key.startswith("synlock#")
        assert "203.0.113.42" not in key
        assert key.count("#") == 2

    def test_missing_subject_skips_locking(self):
        table = MagicMock()

        acquired, _release = acquire_synthesis_lock(
            None, CACHE_KEY, usage_service_factory=lambda: _usage_service(table)
        )

        assert acquired is True
        table.update_item.assert_not_called()


# ========== (b) 엔드포인트 통합 ==========


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _no_rate_limit():
    """합성 엔드포인트의 분당 제한(5/3회)은 이 테스트의 관심사가 아니다"""
    from api.endpoints import synthesis as synthesis_module

    previous = synthesis_module.limiter.enabled
    synthesis_module.limiter.enabled = False
    yield
    synthesis_module.limiter.enabled = previous


@pytest.fixture(autouse=True)
def _no_budget_writes():
    """일 예산 집계는 여기의 관심사가 아니다 (실제 DynamoDB 쓰기 차단)"""
    with patch("api.endpoints.synthesis.record_api_calls"):
        yield


@pytest.fixture
def storage():
    service = MagicMock()
    service.enabled = True
    service.build_cache_key.return_value = CACHE_KEY
    service.get_cached_result.return_value = None  # 캐시 미스
    service.save_user_result.return_value = "https://s3/result.png"
    with patch(
        "api.endpoints.synthesis.get_photo_storage_service", return_value=service
    ):
        yield service


def _synthesis_service(success=True, api_calls=1):
    service = MagicMock()
    service.synthesize_hairstyle.return_value = {
        "success": success,
        "image_base64": "aW1n" if success else None,
        "image_format": "png" if success else None,
        "message": "적용되었습니다." if success else "얼굴을 찾지 못했습니다",
        "api_calls": api_calls,
    }
    service.synthesize_with_reference.return_value = (
        service.synthesize_hairstyle.return_value
    )
    return service


def _post_synthesize(client, headers=None, device_id=None):
    data = {"hairstyle_name": "크롭컷", "gender": "male"}
    if device_id:
        data["device_id"] = device_id
    return client.post(
        "/api/v2/synthesize",
        files={"file": ("face.png", _png_bytes(), "image/png")},
        data=data,
        headers=headers or {},
    )


def _post_reference(client, headers=None):
    return client.post(
        "/api/v2/synthesize-with-reference",
        files={
            "user_photo": ("face.png", _png_bytes(), "image/png"),
            "reference_photo": ("ref.png", _png_bytes(), "image/png"),
        },
        data={"gender": "male"},
        headers=headers or {},
    )


class TestDuplicateRequestRejected:
    def test_second_request_gets_409_without_charging(self, client, storage):
        credit = MagicMock()
        synthesis = _synthesis_service()

        with patch(
            "api.endpoints.synthesis.acquire_synthesis_lock",
            return_value=(False, lambda: None),
        ), patch(
            "api.endpoints.synthesis.get_credit_service", return_value=credit
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service", return_value=synthesis
        ):
            response = _post_synthesize(client, headers=AUTH)

        assert response.status_code == 409
        body = response.json()
        assert body["error"] == "synthesis_in_progress"
        assert body["success"] is False
        # 잠금이 과금보다 앞이므로 크레딧은 건드리지 않는다
        credit.consume.assert_not_called()
        # Gemini 도 호출되지 않는다
        synthesis.synthesize_hairstyle.assert_not_called()

    def test_reference_endpoint_also_rejects(self, client, storage):
        credit = MagicMock()
        synthesis = _synthesis_service()

        with patch(
            "api.endpoints.synthesis.acquire_synthesis_lock",
            return_value=(False, lambda: None),
        ), patch(
            "api.endpoints.synthesis.get_credit_service", return_value=credit
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service", return_value=synthesis
        ):
            response = _post_reference(client, headers=AUTH)

        assert response.status_code == 409
        credit.consume.assert_not_called()
        synthesis.synthesize_with_reference.assert_not_called()

    def test_duplicate_is_logged(self, client, storage):
        with patch(
            "api.endpoints.synthesis.acquire_synthesis_lock",
            return_value=(False, lambda: None),
        ), patch(
            "api.endpoints.synthesis.get_credit_service", return_value=MagicMock()
        ), patch(
            "api.endpoints.synthesis.log_structured"
        ) as mock_log:
            _post_synthesize(client, headers=AUTH)

        failures = [
            call.args[1]
            for call in mock_log.call_args_list
            if call.args[0] == "synthesis_failed"
        ]
        assert failures[-1]["reason"] == "duplicate_in_flight"
        assert failures[-1]["status_code"] == 409
        assert failures[-1]["api_calls"] == 0


class TestLockLifecycle:
    def test_lock_acquired_before_charge_and_released_on_success(self, client, storage):
        credit = MagicMock()
        credit.consume.return_value = 4
        release = MagicMock()
        order = []
        credit.consume.side_effect = lambda *a, **k: order.append("charge") or 4

        def _acquire(*args, **kwargs):
            order.append("lock")
            return True, release

        with patch(
            "api.endpoints.synthesis.acquire_synthesis_lock", side_effect=_acquire
        ), patch(
            "api.endpoints.synthesis.get_credit_service", return_value=credit
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service",
            return_value=_synthesis_service(),
        ), patch(
            "api.endpoints.synthesis.get_user_repository"
        ):
            response = _post_synthesize(client, headers=AUTH)

        assert response.status_code == 200
        assert order == ["lock", "charge"]
        release.assert_called_once()

    def test_lock_released_when_synthesis_fails(self, client, storage):
        credit = MagicMock()
        credit.consume.return_value = 4
        release = MagicMock()

        with patch(
            "api.endpoints.synthesis.acquire_synthesis_lock",
            return_value=(True, release),
        ), patch(
            "api.endpoints.synthesis.get_credit_service", return_value=credit
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service",
            return_value=_synthesis_service(success=False),
        ):
            response = _post_synthesize(client, headers=AUTH)

        assert response.status_code == 422
        release.assert_called_once()
        credit.grant.assert_called_once()  # 환불은 그대로

    def test_lock_released_when_synthesis_raises(self, client, storage):
        credit = MagicMock()
        credit.consume.return_value = 4
        release = MagicMock()
        synthesis = MagicMock()
        synthesis.synthesize_hairstyle.side_effect = RuntimeError("Gemini timeout")

        with patch(
            "api.endpoints.synthesis.acquire_synthesis_lock",
            return_value=(True, release),
        ), patch(
            "api.endpoints.synthesis.get_credit_service", return_value=credit
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service", return_value=synthesis
        ):
            response = _post_synthesize(client, headers=AUTH)

        assert response.status_code == 500
        release.assert_called_once()

    def test_lock_released_when_quota_denies(self, client, storage):
        from services.credit_service import InsufficientCreditsError

        credit = MagicMock()
        credit.consume.side_effect = InsufficientCreditsError(0)
        release = MagicMock()

        with patch(
            "api.endpoints.synthesis.acquire_synthesis_lock",
            return_value=(True, release),
        ), patch("api.endpoints.synthesis.get_credit_service", return_value=credit):
            response = _post_synthesize(client, headers=AUTH)

        assert response.status_code == 402
        release.assert_called_once()


class TestCacheHitExcluded:
    def test_cache_hit_does_not_lock(self, client, storage):
        storage.get_cached_result.return_value = {
            "image_base64": "aW1n",
            "image_format": "png",
        }
        credit = MagicMock()

        with patch(
            "api.endpoints.synthesis.acquire_synthesis_lock"
        ) as mock_acquire, patch(
            "api.endpoints.synthesis.get_credit_service", return_value=credit
        ):
            response = _post_synthesize(client, headers=AUTH)

        assert response.status_code == 200
        assert response.json()["cached"] is True
        mock_acquire.assert_not_called()
        credit.consume.assert_not_called()

    def test_cache_hit_success_log_has_no_api_calls(self, client, storage):
        storage.get_cached_result.return_value = {
            "image_base64": "aW1n",
            "image_format": "png",
        }

        with patch(
            "api.endpoints.synthesis.get_credit_service", return_value=MagicMock()
        ), patch("api.endpoints.synthesis.log_structured") as mock_log:
            _post_synthesize(client, headers=AUTH)

        success = [
            call.args[1]
            for call in mock_log.call_args_list
            if call.args[0] == "synthesis_success"
        ][-1]
        assert success["mode"] == "cache"
        # 캐시는 Gemini 를 부르지 않으므로 비용 집계에서 빠진다
        assert "api_calls" not in success


# ========== (c) 재시도 비용 기록 ==========


class TestApiCallAccounting:
    def test_success_log_carries_api_calls(self, client, storage):
        credit = MagicMock()
        credit.consume.return_value = 4

        with patch(
            "api.endpoints.synthesis.acquire_synthesis_lock",
            return_value=(True, lambda: None),
        ), patch(
            "api.endpoints.synthesis.get_credit_service", return_value=credit
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service",
            return_value=_synthesis_service(api_calls=3),
        ), patch(
            "api.endpoints.synthesis.get_user_repository"
        ), patch(
            "api.endpoints.synthesis.log_structured"
        ) as mock_log:
            response = _post_synthesize(client, headers=AUTH)

        assert response.status_code == 200
        success = [
            call.args[1]
            for call in mock_log.call_args_list
            if call.args[0] == "synthesis_success"
        ][-1]
        assert success["api_calls"] == 3
        # 응답 스키마에는 추가하지 않는다 (구버전 앱 호환)
        assert "api_calls" not in response.json()

    def test_failure_log_carries_api_calls(self, client, storage):
        credit = MagicMock()
        credit.consume.return_value = 4

        with patch(
            "api.endpoints.synthesis.acquire_synthesis_lock",
            return_value=(True, lambda: None),
        ), patch(
            "api.endpoints.synthesis.get_credit_service", return_value=credit
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service",
            return_value=_synthesis_service(success=False, api_calls=3),
        ), patch(
            "api.endpoints.synthesis.log_structured"
        ) as mock_log:
            response = _post_synthesize(client, headers=AUTH)

        assert response.status_code == 422
        failure = [
            call.args[1]
            for call in mock_log.call_args_list
            if call.args[0] == "synthesis_failed"
        ][-1]
        assert failure["reason"] == "synthesis_rejected"
        assert failure["api_calls"] == 3


class TestServiceCountsRealCalls:
    """서비스가 실제 Gemini 호출 횟수를 반환한다 (재시도 횟수 자체는 그대로)"""

    def _service(self, responses):
        from services.hairstyle_synthesis_service import HairstyleSynthesisService

        service = HairstyleSynthesisService()
        client = MagicMock()
        client.models.generate_content.side_effect = responses
        service._client = client
        service.RETRY_DELAY = 0
        return service, client

    def _image_response(self):
        part = MagicMock()
        part.inline_data.data = b"\x89PNG\r\n\x1a\nfake"
        part.inline_data.mime_type = "image/png"
        part.text = None
        response = MagicMock()
        response.candidates = [MagicMock(content=MagicMock(parts=[part]))]
        return response

    def test_single_call_when_first_attempt_succeeds(self):
        service, client = self._service([self._image_response()])

        result = service.synthesize_hairstyle(_png_bytes(), "크롭컷")

        assert result["success"] is True
        assert result["api_calls"] == 1
        assert client.models.generate_content.call_count == 1

    def test_counts_every_retry(self):
        """한 요청이 3배 비용을 낼 수 있다는 것을 로그로 볼 수 있어야 한다"""
        service, client = self._service(
            [Exception("500"), Exception("500"), self._image_response()]
        )

        result = service.synthesize_hairstyle(_png_bytes(), "크롭컷")

        assert result["success"] is True
        assert result["api_calls"] == 3
        assert client.models.generate_content.call_count == 3

    def test_exhausted_retries_report_all_calls(self):
        service, _client = self._service([Exception("500")] * 3)

        result = service.synthesize_hairstyle(_png_bytes(), "크롭컷")

        assert result["success"] is False
        assert result["api_calls"] == 3

    def test_retry_count_itself_is_unchanged(self):
        from services.hairstyle_synthesis_service import HairstyleSynthesisService

        assert HairstyleSynthesisService.MAX_RETRIES == 3

    def test_failure_before_any_call_reports_zero(self):
        service, _client = self._service([self._image_response()])

        result = service.synthesize_hairstyle(b"not-an-image", "크롭컷")

        assert result["success"] is False
        assert result["api_calls"] == 0

    def test_reference_synthesis_counts_calls(self):
        service, client = self._service([Exception("500"), self._image_response()])

        result = service.synthesize_with_reference(_png_bytes(), _png_bytes())

        assert result["success"] is True
        assert result["api_calls"] == 2
        assert client.models.generate_content.call_count == 2
