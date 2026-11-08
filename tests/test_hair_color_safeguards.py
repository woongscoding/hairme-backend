# -*- coding: utf-8 -*-
"""염색 합성 비용 보호 테스트 (스타일 합성 api/endpoints/synthesis.py 와 같은 장치)

- 중복 요청 잠금: 같은 주체 + 같은 사진·컬러가 진행 중이면 두 번째는 과금 전에 409
- 일 예산 상한: 비로그인은 503 (과금·잠금 전), 회원은 통과
- record_api_calls: 실제 Gemini 호출 수를 염색 엔드포인트 이름으로 집계
- 잠금 주체 형식은 synthesis.py 의 _lock_subject 와 같다 (user:/device:)

외부 호출(DynamoDB / Gemini)은 전부 모킹한다.
"""

import io
import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from core.jwt_auth import create_access_token
from core.synthesis_budget import budget_exceeded_response_body
from core.synthesis_lock import (
    NOOP_RELEASE,
    acquire_synthesis_lock,
    duplicate_request_response_body,
    lock_subject,
)
from main import app

DEVICE_ID = "a1b2c3d4e5f60718"
USER_ID = "hair-color-guard-user"
AUTH = {"Authorization": f"Bearer {create_access_token(USER_ID)}"}

SYNTHESIZE = "/api/hair-color/synthesize"
BY_PERSONAL_COLOR = "/api/hair-color/synthesize-by-personal-color"
PATHS = [SYNTHESIZE, BY_PERSONAL_COLOR]
ENDPOINT_NAMES = {
    SYNTHESIZE: "hair-color/synthesize",
    BY_PERSONAL_COLOR: "hair-color/synthesize-by-personal-color",
}


@pytest.fixture(autouse=True)
def _no_rate_limit():
    """분당 제한(5/minute)은 이 테스트의 관심사가 아니다 (리미터는 테스트 간 공유된다)"""
    from api.endpoints import hair_color as hair_color_module

    previous = hair_color_module.limiter.enabled
    hair_color_module.limiter.enabled = False
    yield
    hair_color_module.limiter.enabled = previous


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 90, 60)).save(buf, "PNG")
    return buf.getvalue()


def _color_service(success=True, api_calls=1):
    service = MagicMock()
    service.get_color_by_name.return_value = {"hex": "#8B4513"}
    recommended = MagicMock()
    recommended.name = "밀크브라운"
    recommended.hex = "#C4A484"
    recommended.level = "7"
    recommended.description = ""
    service.get_recommendations.return_value = MagicMock(recommended=[recommended])
    if success:
        service.synthesize_hair_color.return_value = {
            "success": True,
            "image_base64": "aW1n",
            "image_format": "png",
            "message": "적용되었습니다.",
            "api_calls": api_calls,
        }
    else:
        service.synthesize_hair_color.return_value = {
            "success": False,
            "image_base64": None,
            "image_format": None,
            "message": "염색 시뮬레이션 생성에 실패했습니다. 다시 시도해주세요.",
            "api_calls": api_calls,
            "failure_reason": "blocked",
        }
    return service


def _quota_ok():
    """과금 성공 (호출 횟수/환불 여부를 볼 수 있게 목으로)"""
    refund = MagicMock()
    charge = MagicMock(
        return_value=(None, {"mode": "credits", "balance": 4, "tier": "free"}, refund)
    )
    return charge, refund


def _post(client, path, *, member=True, device_id=DEVICE_ID):
    if path == SYNTHESIZE:
        data = {"color_name": "밀크브라운"}
    else:
        data = {"personal_color": "봄웜", "color_index": "0"}
    if device_id:
        data["device_id"] = device_id
    return client.post(
        path,
        files={"file": ("face.png", _png_bytes(), "image/png")},
        data=data,
        headers=AUTH if member else {},
    )


class FakeLockTable:
    """hairstyle_usage 의 조건부 UpdateItem 동작만 흉내낸다 (synlock# 행)"""

    def __init__(self):
        self.items = {}

    def update_item(
        self,
        Key,
        UpdateExpression,
        ExpressionAttributeNames,
        ExpressionAttributeValues,
        ConditionExpression=None,
    ):
        item = self.items.setdefault((Key["device_id"], Key["date"]), {})
        if ConditionExpression is None:  # 해제
            item["lock_expires_at"] = ExpressionAttributeValues[":zero"]
            return {}
        current = item.get("lock_expires_at")
        if current is not None and not current < ExpressionAttributeValues[":now"]:
            from botocore.exceptions import ClientError

            raise ClientError(
                {"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem"
            )
        item["lock_expires_at"] = ExpressionAttributeValues[":expires"]
        return {}


class _Guards:
    """엔드포인트 모듈에 바인딩된 예산/잠금/집계 이름을 한 번에 patch"""

    def __init__(self, *, budget_exceeded=False, lock=None):
        self.table = FakeLockTable()
        usage = MagicMock()
        usage.table = self.table
        self.lock_calls = []
        self.hold_first = False

        def _acquire(subject, cache_key, **kwargs):
            self.lock_calls.append((subject, cache_key, kwargs.get("endpoint")))
            acquired, release = acquire_synthesis_lock(
                subject, cache_key, usage_service_factory=lambda: usage, **kwargs
            )
            if acquired and self.hold_first and len(self.lock_calls) == 1:
                # 첫 요청이 아직 Gemini 를 기다리는 중인 상황 (잠금을 놓지 않음)
                return acquired, NOOP_RELEASE
            return acquired, release

        self.acquire = MagicMock(side_effect=lock or _acquire)
        self.budget = MagicMock(return_value=budget_exceeded)
        self.record = MagicMock()
        self._patches = [
            patch("api.endpoints.hair_color.acquire_synthesis_lock", self.acquire),
            patch("api.endpoints.hair_color.daily_budget_exceeded", self.budget),
            patch("api.endpoints.hair_color.record_api_calls", self.record),
        ]

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()


# ========== 중복 요청 잠금 ==========


class TestDuplicateRequestLock:
    @pytest.mark.parametrize("path", PATHS)
    def test_double_tap_is_charged_once(self, client, path):
        """첫 요청이 진행 중일 때 같은 요청은 과금 전에 409 (중복 과금 방지)"""
        charge, _refund = _quota_ok()
        service = _color_service()

        with _Guards() as guards, patch(
            "api.endpoints.hair_color.charge_synthesis_quota", charge
        ), patch("api.endpoints.hair_color._get_service", return_value=service):
            guards.hold_first = True
            first = _post(client, path)
            second = _post(client, path)

        assert first.status_code == 200, first.text
        assert second.status_code == 409
        assert second.json() == duplicate_request_response_body()
        # 두 번째는 과금도 Gemini 호출도 없다
        assert charge.call_count == 1
        assert service.synthesize_hair_color.call_count == 1
        # 두 요청이 같은 잠금 키를 썼다
        assert guards.lock_calls[0][:2] == guards.lock_calls[1][:2]
        assert guards.lock_calls[0][2] == ENDPOINT_NAMES[path]

    @pytest.mark.parametrize("path", PATHS)
    def test_lock_released_after_completion(self, client, path):
        """끝난 요청의 잠금은 풀려 있어 다음 요청이 정상 과금된다"""
        charge, _refund = _quota_ok()

        with _Guards(), patch(
            "api.endpoints.hair_color.charge_synthesis_quota", charge
        ), patch(
            "api.endpoints.hair_color._get_service", return_value=_color_service()
        ):
            first = _post(client, path)
            second = _post(client, path)

        assert first.status_code == 200
        assert second.status_code == 200
        assert charge.call_count == 2

    def test_lock_released_when_synthesis_fails(self, client):
        charge, refund = _quota_ok()

        with _Guards() as guards, patch(
            "api.endpoints.hair_color.charge_synthesis_quota", charge
        ), patch(
            "api.endpoints.hair_color._get_service",
            return_value=_color_service(success=False),
        ):
            response = _post(client, SYNTHESIZE)

        assert response.status_code == 422
        refund.assert_called_once()
        (item,) = guards.table.items.values()
        assert item["lock_expires_at"] == 0  # 해제됨

    def test_lock_released_when_synthesis_raises(self, client):
        charge, refund = _quota_ok()
        service = _color_service()
        service.synthesize_hair_color.side_effect = RuntimeError("gemini down")

        with _Guards() as guards, patch(
            "api.endpoints.hair_color.charge_synthesis_quota", charge
        ), patch("api.endpoints.hair_color._get_service", return_value=service):
            response = _post(client, SYNTHESIZE)

        assert response.status_code == 500
        refund.assert_called_once()
        (item,) = guards.table.items.values()
        assert item["lock_expires_at"] == 0

    def test_lock_released_when_quota_denied(self, client):
        from fastapi.responses import JSONResponse

        denied = JSONResponse(status_code=402, content={"error": "no_credits"})
        charge = MagicMock(return_value=(denied, None, None))

        with _Guards() as guards, patch(
            "api.endpoints.hair_color.charge_synthesis_quota", charge
        ), patch(
            "api.endpoints.hair_color._get_service", return_value=_color_service()
        ):
            response = _post(client, SYNTHESIZE)

        assert response.status_code == 402
        (item,) = guards.table.items.values()
        assert item["lock_expires_at"] == 0

    def test_different_color_is_not_a_duplicate(self, client):
        charge, _refund = _quota_ok()

        with _Guards() as guards, patch(
            "api.endpoints.hair_color.charge_synthesis_quota", charge
        ), patch(
            "api.endpoints.hair_color._get_service", return_value=_color_service()
        ):
            guards.hold_first = True
            first = _post(client, SYNTHESIZE)
            second = client.post(
                SYNTHESIZE,
                files={"file": ("face.png", _png_bytes(), "image/png")},
                data={"color_name": "애쉬브라운"},
                headers=AUTH,
            )

        assert first.status_code == 200
        assert second.status_code == 200
        assert charge.call_count == 2

    @pytest.mark.parametrize("path", PATHS)
    def test_lock_subject_format(self, client, path):
        """회원은 user:<id>, 비로그인은 device:<id> (synthesis.py 와 같은 형식)"""
        charge, _refund = _quota_ok()

        with _Guards() as guards, patch(
            "api.endpoints.hair_color.charge_synthesis_quota", charge
        ), patch(
            "api.endpoints.hair_color._get_service", return_value=_color_service()
        ):
            _post(client, path, member=True)
            _post(client, path, member=False, device_id=f"  {DEVICE_ID} ")

        assert guards.lock_calls[0][0] == f"user:{USER_ID}"
        assert guards.lock_calls[1][0] == f"device:{DEVICE_ID}"

    def test_lock_subject_matches_synthesis_endpoint(self):
        from api.endpoints.synthesis import _lock_subject

        for user_id, device_id in [
            (USER_ID, None),
            (USER_ID, DEVICE_ID),
            (None, DEVICE_ID),
            (None, f"  {DEVICE_ID}  "),
            (None, "   "),
            (None, None),
        ]:
            assert lock_subject(user_id, device_id) == _lock_subject(user_id, device_id)


# ========== 일 예산 상한 ==========


class TestDailyBudget:
    @pytest.mark.parametrize("path", PATHS)
    def test_anonymous_blocked_with_503_before_charge_and_lock(self, client, path):
        charge, _refund = _quota_ok()
        service = _color_service()

        with _Guards(budget_exceeded=True) as guards, patch(
            "api.endpoints.hair_color.charge_synthesis_quota", charge
        ), patch("api.endpoints.hair_color._get_service", return_value=service):
            response = _post(client, path, member=False)

        assert response.status_code == 503
        assert response.json() == budget_exceeded_response_body()
        guards.budget.assert_called_once_with(endpoint=ENDPOINT_NAMES[path])
        # 거절될 요청은 잠금도, 과금도, Gemini 호출도, 집계도 하지 않는다
        guards.acquire.assert_not_called()
        charge.assert_not_called()
        service.synthesize_hair_color.assert_not_called()
        guards.record.assert_not_called()

    @pytest.mark.parametrize("path", PATHS)
    def test_member_passes_even_when_budget_exceeded(self, client, path):
        """synthesis.py 와 같다 - 상한은 비로그인만 막는다"""
        charge, _refund = _quota_ok()

        with _Guards(budget_exceeded=True) as guards, patch(
            "api.endpoints.hair_color.charge_synthesis_quota", charge
        ), patch(
            "api.endpoints.hair_color._get_service", return_value=_color_service()
        ):
            response = _post(client, path, member=True)

        assert response.status_code == 200
        guards.budget.assert_not_called()
        charge.assert_called_once()


# ========== 실제 Gemini 호출 수 집계 ==========


class TestApiCallAccounting:
    @pytest.mark.parametrize("path", PATHS)
    def test_success_records_calls_with_endpoint_name(self, client, path):
        charge, _refund = _quota_ok()

        with _Guards() as guards, patch(
            "api.endpoints.hair_color.charge_synthesis_quota", charge
        ), patch(
            "api.endpoints.hair_color._get_service",
            return_value=_color_service(api_calls=2),
        ):
            response = _post(client, path)

        assert response.status_code == 200
        guards.record.assert_called_once_with(2, endpoint=ENDPOINT_NAMES[path])

    @pytest.mark.parametrize("path", PATHS)
    def test_failure_also_records_calls(self, client, path):
        """실패해도 비용은 나갔으므로 집계한다"""
        charge, _refund = _quota_ok()

        with _Guards() as guards, patch(
            "api.endpoints.hair_color.charge_synthesis_quota", charge
        ), patch(
            "api.endpoints.hair_color._get_service",
            return_value=_color_service(success=False, api_calls=1),
        ):
            response = _post(client, path)

        assert response.status_code == 422
        guards.record.assert_called_once_with(1, endpoint=ENDPOINT_NAMES[path])

    def test_duplicate_rejection_does_not_record(self, client):
        charge, _refund = _quota_ok()

        with _Guards(lock=lambda *a, **k: (False, NOOP_RELEASE)) as guards, patch(
            "api.endpoints.hair_color.charge_synthesis_quota", charge
        ), patch(
            "api.endpoints.hair_color._get_service", return_value=_color_service()
        ):
            response = _post(client, SYNTHESIZE)

        assert response.status_code == 409
        guards.record.assert_not_called()
        charge.assert_not_called()

    def test_success_log_carries_api_calls(self, client):
        charge, _refund = _quota_ok()
        events = []

        with _Guards(), patch(
            "api.endpoints.hair_color.charge_synthesis_quota", charge
        ), patch(
            "api.endpoints.hair_color._get_service",
            return_value=_color_service(api_calls=3),
        ), patch(
            "api.endpoints.hair_color.log_structured",
            side_effect=lambda name, data: events.append((name, data)),
        ):
            _post(client, SYNTHESIZE)

        success = [d for n, d in events if n == "hair_color_synthesis_success"]
        assert success and success[0]["api_calls"] == 3
        assert success[0]["endpoint"] == "hair-color/synthesize"


# ========== 서비스가 api_calls 를 보고한다 ==========


class TestHairColorServiceReportsApiCalls:
    def _service(self, responses):
        from services.hair_color_service import HairColorService

        service = HairColorService()
        client = MagicMock()
        client.models.generate_content.side_effect = responses
        service._gemini_client = client
        return service, client

    @staticmethod
    def _image_response():
        part = MagicMock()
        part.inline_data.data = b"\x89PNG\r\n\x1a\nfake"
        part.inline_data.mime_type = "image/png"
        part.text = None
        response = MagicMock()
        response.candidates = [MagicMock(content=MagicMock(parts=[part]))]
        return response

    def test_success_reports_one_call(self):
        service, _client = self._service([self._image_response()])

        result = service.synthesize_hair_color(_png_bytes(), "밀크브라운", "#C4A484")

        assert result["success"] is True
        assert result["api_calls"] == 1

    def test_retries_are_counted(self):
        from google.genai import errors

        server_error = errors.ServerError(
            503, {"error": {"code": 503, "message": "x", "status": "UNAVAILABLE"}}
        )
        service, client = self._service([server_error, self._image_response()])

        with patch("services.gemini_image_config.time.sleep"):
            result = service.synthesize_hair_color(
                _png_bytes(), "밀크브라운", "#C4A484"
            )

        assert result["success"] is True
        assert result["api_calls"] == 2
        assert client.models.generate_content.call_count == 2

    def test_failure_before_any_call_reports_zero(self):
        service, client = self._service([self._image_response()])

        result = service.synthesize_hair_color(b"not-an-image", "밀크브라운", "#C4A484")

        assert result["success"] is False
        assert result["api_calls"] == 0
        client.models.generate_content.assert_not_called()
