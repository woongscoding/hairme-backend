# -*- coding: utf-8 -*-
"""합성 공통 파이프라인 테스트 (services/synthesis_pipeline.py)

돈이 걸린 규칙을 두 엔드포인트(스타일 합성 / 염색 합성) 모두에서 확인한다:
- 과금 뒤 실패(422 거절, Gemini 예외, 저장 예외, 422 처리 중 예외)는 정확히 1회 환불
- 과금 전에 끝나는 실패(검증 400, 예산 503, 잠금 409)는 차감도 환불도 없음
- 사전 거절 로그에는 api_calls 가 없고, 실패 로그 이벤트 이름은 엔드포인트마다 그대로

과금은 실제 core.quota.charge_synthesis_quota(_refund_once 포함)를 타고,
크레딧 저장소만 가짜로 바꾼다. 외부 호출(DynamoDB / S3 / Gemini)은 전부 모킹한다.
"""

import io
import json
import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from core.jwt_auth import create_access_token
from core.quota import charge_synthesis_quota
from core.synthesis_budget import budget_exceeded_response_body
from core.synthesis_lock import NOOP_RELEASE, duplicate_request_response_body
from main import app
from services.synthesis_pipeline import (
    INTERNAL_ERROR_DETAIL,
    SynthesisPipeline,
    charge_mode,
    refund_on_error,
    synthesis_rejected_body,
)

USER_ID = "pipeline-user"
DEVICE_ID = "a1b2c3d4e5f60718"
AUTH = {"Authorization": f"Bearer {create_access_token(USER_ID)}"}


# ========== 공용 가짜 ==========


class FakeCredits:
    """크레딧 지갑: 차감/환불 횟수만 센다"""

    def __init__(self, tier="free"):
        self.tier = tier
        self.consumed = 0
        self.granted = []

    def get_wallet(self, user_id):
        return {"balance": 5, "paid_balance": 0}

    def consume_for_synthesis(self, user_id, cost):
        self.consumed += 1
        return 4, self.tier

    def grant(self, user_id, amount, reason=None, paid=False):
        self.granted.append({"amount": amount, "reason": reason, "paid": paid})


def _pipeline(**overrides):
    events = []
    deps = dict(
        endpoint="unit",
        failed_event="unit_failed",
        user_id=None,
        start_time=0.0,
        log=lambda name, data: events.append((name, data)),
        budget_exceeded=MagicMock(return_value=False),
        acquire_lock=MagicMock(return_value=(True, MagicMock())),
        record_calls=MagicMock(),
    )
    deps.update(overrides)
    return SynthesisPipeline(**deps), events


def _member_charge(credits):
    """실제 과금 함수 (환불은 _refund_once 로 감싸진다)"""
    return charge_synthesis_quota(
        USER_ID, None, None, endpoint="unit", credit_service_factory=lambda: credits
    )


# ========== 단위: 공통 헬퍼 ==========


class TestHelpers:
    def test_charge_mode(self):
        assert charge_mode(USER_ID) == "credits"
        assert charge_mode(None) == "device"
        # 차감 결과가 있으면 그 mode 가 우선
        assert charge_mode(USER_ID, {"mode": "device"}) == "device"
        assert charge_mode(None, {}) == "device"

    def test_rejected_body_keeps_contract_and_extra_position(self):
        body = synthesis_rejected_body(
            "실패", 1.5, {"mode": "credits", "balance": 4}, color_name="밀크브라운"
        )
        assert list(body) == [
            "success",
            "error",
            "message",
            "color_name",
            "processing_time",
            "refunded",
            "balance",
        ]
        assert body["success"] is False
        assert body["error"] == "synthesis_rejected"
        assert body["refunded"] is True
        # 환불 후 잔액 = 차감 직후 잔액 + 1회분
        assert body["balance"] == 5

    def test_rejected_body_balance_is_null_for_device_flow(self):
        body = synthesis_rejected_body("실패", 1.0, {"mode": "device"})
        assert body["balance"] is None


class TestRefundOnError:
    def test_refunds_and_reraises(self):
        refund = MagicMock()
        with pytest.raises(RuntimeError):
            with refund_on_error(refund):
                raise RuntimeError("gemini timeout")
        refund.assert_called_once()

    def test_no_refund_without_error(self):
        refund = MagicMock()
        with refund_on_error(refund):
            pass
        refund.assert_not_called()

    def test_real_quota_refund_runs_once_even_if_called_twice(self):
        """422 로 환불한 뒤 같은 블록에서 예외가 나도 실제 환불은 1회"""
        credits = FakeCredits()
        _error, quota, refund = _member_charge(credits)
        pipeline, _events = _pipeline(user_id=USER_ID)

        with pytest.raises(RuntimeError):
            with refund_on_error(refund):
                pipeline.reject_synthesis(
                    {"success": False, "message": "실패"}, quota, refund, 1, 0.5
                )
                raise RuntimeError("422 응답 만든 뒤 예외")

        assert credits.consumed == 1
        assert len(credits.granted) == 1
        assert credits.granted[0]["reason"] == "refund"


# ========== 단위: 과금 전 가드 ==========


class TestPreChargeGuard:
    def test_anonymous_over_budget_is_503_before_lock(self):
        acquire = MagicMock()
        pipeline, events = _pipeline(
            budget_exceeded=MagicMock(return_value=True), acquire_lock=acquire
        )

        rejection, release = pipeline.pre_charge_guard(
            DEVICE_ID, "k" * 64, log_fields={"cache_key": "k" * 16}
        )

        assert rejection.status_code == 503
        assert json.loads(rejection.body) == budget_exceeded_response_body()
        acquire.assert_not_called()
        assert release is NOOP_RELEASE
        name, data = events[-1]
        assert name == "unit_failed"
        assert data["reason"] == "daily_budget_exceeded"
        assert data["status_code"] == 503
        assert data["cache_key"] == "k" * 16
        assert data["authenticated"] is False
        assert "api_calls" not in data

    def test_member_skips_budget(self):
        budget = MagicMock(return_value=True)
        release = MagicMock()
        pipeline, events = _pipeline(
            user_id=USER_ID,
            budget_exceeded=budget,
            acquire_lock=MagicMock(return_value=(True, release)),
        )

        rejection, got_release = pipeline.pre_charge_guard(None, "k")

        assert rejection is None
        assert got_release is release
        budget.assert_not_called()
        assert events == []

    def test_lock_subject_and_endpoint(self):
        acquire = MagicMock(return_value=(True, MagicMock()))
        pipeline, _ = _pipeline(acquire_lock=acquire)
        pipeline.pre_charge_guard(f"  {DEVICE_ID} ", "key")
        acquire.assert_called_once_with(f"device:{DEVICE_ID}", "key", endpoint="unit")

        acquire.reset_mock()
        member, _ = _pipeline(user_id=USER_ID, acquire_lock=acquire)
        member.pre_charge_guard(DEVICE_ID, "key")
        assert acquire.call_args.args[0] == f"user:{USER_ID}"

    def test_duplicate_is_409(self):
        pipeline, events = _pipeline(
            user_id=USER_ID, acquire_lock=MagicMock(return_value=(False, MagicMock()))
        )

        rejection, release = pipeline.pre_charge_guard(None, "key")

        assert rejection.status_code == 409
        assert json.loads(rejection.body) == duplicate_request_response_body()
        assert release is NOOP_RELEASE
        name, data = events[-1]
        assert name == "unit_failed"
        assert data["reason"] == "duplicate_in_flight"
        assert data["authenticated"] is True
        assert "api_calls" not in data


class TestRecordAndReject:
    def test_record_gemini_calls(self):
        record = MagicMock()
        pipeline, _ = _pipeline(record_calls=record)
        assert pipeline.record_gemini_calls({"api_calls": 3}) == 3
        record.assert_called_once_with(3, endpoint="unit")
        assert pipeline.record_gemini_calls({}) == 0

    def test_reject_refunds_then_logs(self):
        order = []
        refund = MagicMock(side_effect=lambda: order.append("refund"))
        pipeline, events = _pipeline(
            user_id=USER_ID,
            log=lambda name, data: order.append(name) or events.append((name, data)),
        )

        response = pipeline.reject_synthesis(
            {"success": False, "message": "AI가 잠깐 졸았나봐요"},
            {"mode": "credits", "balance": 4},
            refund,
            2,
            3.2,
            log_fields={"cache_key": "abc"},
        )

        assert response.status_code == 422
        assert order == ["refund", "unit_failed"]
        _name, data = events[-1]
        assert data["reason"] == "synthesis_rejected"
        assert data["api_calls"] == 2
        assert data["refunded"] is True
        assert data["cache_key"] == "abc"
        assert data["mode"] == "credits"


# ========== 통합: 두 엔드포인트의 환불 1회 / 차감 전 실패 무환불 ==========


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def _no_rate_limit():
    from api.endpoints import hair_color, synthesis

    previous = (synthesis.limiter.enabled, hair_color.limiter.enabled)
    synthesis.limiter.enabled = False
    hair_color.limiter.enabled = False
    yield
    synthesis.limiter.enabled, hair_color.limiter.enabled = previous


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def _result(success=True, api_calls=1):
    if success:
        return {
            "success": True,
            "image_base64": "aW1n",
            "image_format": "png",
            "message": "적용되었습니다.",
            "api_calls": api_calls,
        }
    return {
        "success": False,
        "image_base64": None,
        "image_format": None,
        "message": "AI가 잠깐 졸았나봐요.. 다시 한번 눌러주세요!",
        "api_calls": api_calls,
        "failure_reason": "blocked",
    }


class _Endpoint:
    """엔드포인트별 patch 경로와 요청 모양만 다르다"""

    def __init__(self, module, path, failed_event, form):
        self.module = module
        self.path = path
        self.failed_event = failed_event
        self.form = form

    def post(self, client, *, headers=AUTH, filename="face.png", device_id=None):
        data = dict(self.form)
        if device_id:
            data["device_id"] = device_id
        return client.post(
            self.path,
            files={"file": (filename, _png_bytes(), "image/png")},
            data=data,
            headers=headers,
        )


SYNTH = _Endpoint(
    "api.endpoints.synthesis",
    "/api/v2/synthesize",
    "synthesis_failed",
    {"hairstyle_name": "레이어드 컷", "gender": "male"},
)
COLOR = _Endpoint(
    "api.endpoints.hair_color",
    "/api/hair-color/synthesize",
    "hair_color_synthesis_failed",
    {"color_name": "밀크브라운"},
)


class _World:
    """한 엔드포인트의 외부 의존성을 전부 가짜로 바꾼다"""

    def __init__(self, endpoint, *, result=None, service_error=None, acquired=True):
        self.endpoint = endpoint
        self.credits = FakeCredits()
        self.usage = MagicMock()
        self.release = MagicMock()
        self.log = MagicMock()
        self.record = MagicMock()
        self.storage = MagicMock()
        self.storage.enabled = True
        self.storage.build_cache_key.return_value = "c" * 64
        self.storage.get_cached_result.return_value = None
        self.service = MagicMock()
        self.service.get_color_by_name.return_value = {"hex": "#8B4513"}
        call = (
            self.service.synthesize_hairstyle
            if endpoint is SYNTH
            else self.service.synthesize_hair_color
        )
        if service_error is not None:
            call.side_effect = service_error
        else:
            call.return_value = result or _result()
        self.acquired = acquired
        self._patches = []

    def __enter__(self):
        m = self.endpoint.module
        targets = {
            f"{m}.daily_budget_exceeded": MagicMock(return_value=False),
            f"{m}.acquire_synthesis_lock": MagicMock(
                return_value=(self.acquired, self.release)
            ),
            f"{m}.record_api_calls": self.record,
            f"{m}.log_structured": self.log,
            # 염색은 core.quota 의 기본 팩토리, 스타일 합성은 모듈 이름을 넘긴다
            "core.quota.get_credit_service": lambda: self.credits,
            "core.quota.get_usage_limit_service": lambda: self.usage,
        }
        if self.endpoint is SYNTH:
            targets.update(
                {
                    f"{m}.get_credit_service": lambda: self.credits,
                    f"{m}.get_usage_limit_service": lambda: self.usage,
                    f"{m}.get_photo_storage_service": lambda: self.storage,
                    f"{m}.get_synthesis_service": lambda: self.service,
                    f"{m}.get_user_repository": MagicMock(),
                    f"{m}._safe_product_recommendations": MagicMock(return_value=[]),
                }
            )
        else:
            targets[f"{m}._get_service"] = lambda: self.service
        for target, value in targets.items():
            p = patch(target, value)
            p.start()
            self._patches.append(p)
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()

    def patch(self, name, value):
        p = patch(f"{self.endpoint.module}.{name}", value)
        p.start()
        self._patches.append(p)

    def failures(self):
        return [
            c.args[1]
            for c in self.log.call_args_list
            if c.args[0] == self.endpoint.failed_event
        ]


ENDPOINTS = pytest.mark.parametrize("endpoint", [SYNTH, COLOR], ids=["style", "color"])


@ENDPOINTS
class TestRefundExactlyOnce:
    def test_success_charges_once_without_refund(self, client, endpoint):
        with _World(endpoint) as world:
            response = endpoint.post(client)

        assert response.status_code == 200
        assert world.credits.consumed == 1
        assert world.credits.granted == []
        world.record.assert_called_once()
        world.release.assert_called_once()

    def test_gemini_rejection_refunds_once(self, client, endpoint):
        with _World(endpoint, result=_result(success=False, api_calls=2)) as world:
            response = endpoint.post(client)

        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "synthesis_rejected"
        assert body["refunded"] is True
        assert body["balance"] == 5
        assert world.credits.consumed == 1
        assert len(world.credits.granted) == 1
        # 실패한 호출도 비용은 나갔으므로 집계한다
        assert world.record.call_args.args[0] == 2
        assert world.failures()[-1]["reason"] == "synthesis_rejected"
        assert world.failures()[-1]["api_calls"] == 2
        world.release.assert_called_once()

    def test_gemini_exception_refunds_once(self, client, endpoint):
        with _World(endpoint, service_error=TimeoutError("gemini timeout")) as world:
            response = endpoint.post(client)

        assert response.status_code == 500
        assert response.json()["detail"] == INTERNAL_ERROR_DETAIL
        assert world.credits.consumed == 1
        assert len(world.credits.granted) == 1
        world.release.assert_called_once()

    def test_exception_after_rejection_still_refunds_once(self, client, endpoint):
        """422 경로에서 환불한 뒤 예외가 나도(로그 실패 등) 두 번 환불하지 않는다"""
        world = _World(endpoint, result=_result(success=False))
        raised = []

        def flaky_log(name, data):
            if data.get("reason") == "synthesis_rejected" and not raised:
                raised.append(True)
                raise RuntimeError("log sink down")

        with world:
            world.log.side_effect = flaky_log
            response = endpoint.post(client)

        assert response.status_code == 500
        assert raised
        assert world.credits.consumed == 1
        assert len(world.credits.granted) == 1


class TestStyleOnlyPaths:
    def test_storage_exception_after_success_refunds_once(self, client):
        """스타일 합성은 저장까지 환불 보호 범위 안이다 (결과를 못 넘기면 환불)"""
        with _World(SYNTH) as world:
            world.storage.save_cached_result.side_effect = RuntimeError("s3 down")
            response = SYNTH.post(client)

        assert response.status_code == 500
        assert world.credits.consumed == 1
        assert len(world.credits.granted) == 1

    def test_rejection_log_uses_charged_tier_cache_key(self, client):
        """유료 차감으로 등급이 바뀌면 422 로그의 cache_key 도 바뀐 키다"""
        with _World(SYNTH, result=_result(success=False)) as world:
            world.credits.tier = "paid"
            world.storage.build_cache_key.side_effect = lambda *parts: (
                "p" * 64 if parts[-1] == b"tier:paid" else "f" * 64
            )
            response = SYNTH.post(client)

        assert response.status_code == 422
        assert world.failures()[-1]["cache_key"] == "p" * 16
        # 유료분 차감은 유료분으로 되돌린다
        assert world.credits.granted == [
            {"amount": 1, "reason": "refund", "paid": True}
        ]

    def test_cache_hit_skips_charge_lock_and_budget(self, client):
        with _World(SYNTH) as world:
            world.storage.get_cached_result.return_value = {
                "image_base64": "aW1n",
                "image_format": "png",
            }
            response = SYNTH.post(client)

        assert response.status_code == 200
        assert response.json()["cached"] is True
        assert world.credits.consumed == 0
        world.release.assert_not_called()
        world.record.assert_not_called()


@ENDPOINTS
class TestNoChargeNoRefundBeforeCharge:
    def test_invalid_upload_is_400_without_charge(self, client, endpoint):
        with _World(endpoint) as world:
            response = endpoint.post(client, filename="face.gif")

        assert response.status_code == 400
        assert response.json()["error"] == "invalid_file_format"
        assert world.credits.consumed == 0
        assert world.credits.granted == []

    def test_duplicate_is_409_without_charge(self, client, endpoint):
        with _World(endpoint, acquired=False) as world:
            response = endpoint.post(client)

        assert response.status_code == 409
        assert response.json() == duplicate_request_response_body()
        assert world.credits.consumed == 0
        assert world.credits.granted == []
        world.record.assert_not_called()
        failure = world.failures()[-1]
        assert failure["reason"] == "duplicate_in_flight"
        assert "api_calls" not in failure

    def test_anonymous_budget_is_503_without_charge(self, client, endpoint):
        with _World(endpoint) as world:
            world.patch("daily_budget_exceeded", MagicMock(return_value=True))
            response = endpoint.post(client, headers={}, device_id=DEVICE_ID)

        assert response.status_code == 503
        assert response.json() == budget_exceeded_response_body()
        # 무료 한도 카운터도 IP 카운터도 건드리지 않는다
        world.usage.check_and_increment_usage.assert_not_called()
        world.usage.increment_daily_counter.assert_not_called()
        world.usage.decrement_usage.assert_not_called()
        failure = world.failures()[-1]
        assert failure["reason"] == "daily_budget_exceeded"
        assert failure["authenticated"] is False
        assert "api_calls" not in failure

    def test_insufficient_credits_has_no_refund(self, client, endpoint):
        from services.credit_service import InsufficientCreditsError

        with _World(endpoint) as world:
            world.credits.consume_for_synthesis = MagicMock(
                side_effect=InsufficientCreditsError(0)
            )
            world.credits.grant = MagicMock()
            response = endpoint.post(client)

        assert response.status_code == 402
        world.credits.grant.assert_not_called()
        world.service.synthesize_hairstyle.assert_not_called()
        world.service.synthesize_hair_color.assert_not_called()
        world.release.assert_called_once()
