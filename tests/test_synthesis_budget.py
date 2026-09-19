# -*- coding: utf-8 -*-
"""일 합성 예산 상한 테스트 (DAILY_SYNTHESIS_BUDGET)

- 기본값 0 은 무제한 (집계만 하고 막지 않는다)
- 하루 전체 실제 Gemini 호출 수(api_calls 합계, 캐시 히트 제외)를
  hairstyle_usage 의 budget# 네임스페이스에 UpdateItem 으로 원자 증가시켜 집계
- 상한 도달 시 비로그인 합성만 503 + 안내 문구로 거절, 회원은 전원 통과
- synthesis_failed 에 reason=daily_budget_exceeded 로 남긴다
- 집계 저장소 장애 시에는 통과시킨다 (fail-open)
- 확인은 잠금 획득보다 먼저 한다

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

from config.settings import settings
from core.jwt_auth import create_access_token
from core.synthesis_budget import (
    BUDGET_KEY,
    budget_exceeded_response_body,
    daily_budget_exceeded,
    record_api_calls,
)
from main import app

DEVICE_ID = "a1b2c3d4e5f60718"
USER_ID = "budget-test-user"
CACHE_KEY = "e" * 64
AUTH = {"Authorization": f"Bearer {create_access_token(USER_ID)}"}


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def _usage_service(table=None):
    service = MagicMock()
    service.table = table if table is not None else MagicMock()
    return service


def _table_with_count(count):
    table = MagicMock()
    table.get_item.return_value = {} if count is None else {"Item": {"count": count}}
    return table


@pytest.fixture
def budget_limit(monkeypatch):
    """상한 10 으로 설정"""
    monkeypatch.setattr(settings, "DAILY_SYNTHESIS_BUDGET", 10)
    return 10


# ========== (a) 예산 모듈 ==========


class TestDailyBudgetCheck:
    def test_default_is_unlimited(self):
        """기본값 0 은 무제한 - 저장소를 읽지도 않는다"""
        assert settings.DAILY_SYNTHESIS_BUDGET == 0

        table = MagicMock()
        assert (
            daily_budget_exceeded(usage_service_factory=lambda: _usage_service(table))
            is False
        )
        table.get_item.assert_not_called()

    def test_zero_limit_skips_lookup(self, monkeypatch):
        monkeypatch.setattr(settings, "DAILY_SYNTHESIS_BUDGET", 0)
        table = MagicMock()

        assert (
            daily_budget_exceeded(usage_service_factory=lambda: _usage_service(table))
            is False
        )
        table.get_item.assert_not_called()

    def test_reads_todays_budget_row(self, budget_limit):
        table = _table_with_count(3)

        daily_budget_exceeded(usage_service_factory=lambda: _usage_service(table))

        key = table.get_item.call_args.kwargs["Key"]
        assert key["device_id"] == BUDGET_KEY
        # 정렬 키는 KST 날짜 (hairstyle_usage 의 기존 규칙)
        assert len(key["date"]) == len("2026-09-19")
        assert key["date"].count("-") == 2

    @pytest.mark.parametrize(
        "used,expected", [(0, False), (9, False), (10, True), (23, True)]
    )
    def test_threshold(self, budget_limit, used, expected):
        table = _table_with_count(used)

        assert (
            daily_budget_exceeded(usage_service_factory=lambda: _usage_service(table))
            is expected
        )

    def test_missing_row_counts_as_zero(self, budget_limit):
        table = _table_with_count(None)

        assert (
            daily_budget_exceeded(usage_service_factory=lambda: _usage_service(table))
            is False
        )

    def test_store_failure_passes_through(self, budget_limit):
        table = MagicMock()
        table.get_item.side_effect = Exception("DynamoDB down")

        with patch("core.synthesis_budget.log_structured") as mock_log:
            allowed = daily_budget_exceeded(
                endpoint="synthesize",
                usage_service_factory=lambda: _usage_service(table),
            )

        assert allowed is False  # 합성을 막지 않는다
        event_type, data = mock_log.call_args.args
        assert event_type == "synthesis_budget_unavailable"
        assert data["operation"] == "read"


class TestRecordApiCalls:
    def test_atomic_increment_by_api_calls(self):
        table = MagicMock()

        record_api_calls(3, usage_service_factory=lambda: _usage_service(table))

        kwargs = table.update_item.call_args.kwargs
        assert kwargs["Key"]["device_id"] == BUDGET_KEY
        assert kwargs["ExpressionAttributeValues"][":inc"] == 3
        # 조건 없는 원자 증가 - 상한을 넘겨도 실제 호출 수를 기록한다
        assert "ConditionExpression" not in kwargs
        assert "if_not_exists" in kwargs["UpdateExpression"]
        assert kwargs["ExpressionAttributeNames"]["#cnt"] == "count"
        # expire_at(TTL)은 남은 행 청소용이다. 날짜가 바뀌면 정렬 키가 바뀌어
        # 새 행에서 0 부터 시작하므로, 집계 초기화는 TTL 삭제와 무관하다
        # (DynamoDB TTL 삭제는 즉시가 아니라 최대 48시간까지 늦어질 수 있다)
        assert kwargs["ExpressionAttributeValues"][":ttl"] > 0

    def test_zero_or_negative_is_not_recorded(self):
        table = MagicMock()

        record_api_calls(0, usage_service_factory=lambda: _usage_service(table))
        record_api_calls(-1, usage_service_factory=lambda: _usage_service(table))

        table.update_item.assert_not_called()

    def test_failure_is_swallowed(self):
        table = MagicMock()
        table.update_item.side_effect = Exception("DynamoDB down")

        with patch("core.synthesis_budget.log_structured") as mock_log:
            record_api_calls(
                2,
                endpoint="synthesize",
                usage_service_factory=lambda: _usage_service(table),
            )

        assert mock_log.call_args.args[1]["operation"] == "record"

    def test_budget_key_uses_server_namespace(self):
        """클라이언트 device_id 는 '#' 이 금지되어 키가 겹치지 않는다"""
        from services.usage_limit_service import validate_device_id

        assert BUDGET_KEY.startswith("budget#")
        with pytest.raises(ValueError):
            validate_device_id(BUDGET_KEY)


# ========== (b) 엔드포인트 통합 ==========


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _no_rate_limit():
    from api.endpoints import synthesis as synthesis_module

    previous = synthesis_module.limiter.enabled
    synthesis_module.limiter.enabled = False
    yield
    synthesis_module.limiter.enabled = previous


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
    result = {
        "success": success,
        "image_base64": "aW1n" if success else None,
        "image_format": "png" if success else None,
        "message": "적용되었습니다." if success else "얼굴을 찾지 못했습니다",
        "api_calls": api_calls,
    }
    service.synthesize_hairstyle.return_value = result
    service.synthesize_with_reference.return_value = result
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


def _post_reference(client, headers=None, device_id=None):
    data = {"gender": "male"}
    if device_id:
        data["device_id"] = device_id
    return client.post(
        "/api/v2/synthesize-with-reference",
        files={
            "user_photo": ("face.png", _png_bytes(), "image/png"),
            "reference_photo": ("ref.png", _png_bytes(), "image/png"),
        },
        data=data,
        headers=headers or {},
    )


class FakeUsageService:
    """비로그인 일일 한도를 통과시키는 사용량 목"""

    def __init__(self):
        self.table = MagicMock()

    def increment_daily_counter(self, key, limit):
        return True

    def decrement_daily_counter(self, key):
        pass

    def check_and_increment_usage(self, device_id):
        return {"allowed": True, "daily_limit": 3, "used": 1, "remaining": 2}

    def decrement_usage(self, device_id):
        pass


class TestBudgetBlocksAnonymousOnly:
    def test_anonymous_rejected_with_503(self, client, storage):
        synthesis = _synthesis_service()

        with patch(
            "api.endpoints.synthesis.daily_budget_exceeded", return_value=True
        ), patch("api.endpoints.synthesis.acquire_synthesis_lock") as mock_lock, patch(
            "api.endpoints.synthesis.get_usage_limit_service",
            return_value=FakeUsageService(),
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service", return_value=synthesis
        ):
            response = _post_synthesize(client, device_id=DEVICE_ID)

        assert response.status_code == 503
        body = response.json()
        assert body["error"] == "daily_budget_exceeded"
        assert body["success"] is False
        assert body["message"]  # 안내 문구가 비어있지 않다
        # 확인은 잠금 획득보다 먼저 - 거절된 요청은 잠금을 잡지 않는다
        mock_lock.assert_not_called()
        synthesis.synthesize_hairstyle.assert_not_called()

    def test_member_passes_even_when_budget_exhausted(self, client, storage):
        credit = MagicMock()
        credit.consume.return_value = 4
        synthesis = _synthesis_service()
        budget_check = MagicMock(return_value=True)

        with patch(
            "api.endpoints.synthesis.daily_budget_exceeded", budget_check
        ), patch(
            "api.endpoints.synthesis.acquire_synthesis_lock",
            return_value=(True, lambda: None),
        ), patch(
            "api.endpoints.synthesis.get_credit_service", return_value=credit
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service", return_value=synthesis
        ), patch(
            "api.endpoints.synthesis.record_api_calls"
        ), patch(
            "api.endpoints.synthesis.get_user_repository"
        ):
            response = _post_synthesize(client, headers=AUTH)

        assert response.status_code == 200
        # 회원은 상한 확인 자체를 하지 않는다
        budget_check.assert_not_called()
        synthesis.synthesize_hairstyle.assert_called_once()

    def test_anonymous_passes_below_limit(self, client, storage):
        synthesis = _synthesis_service()

        with patch(
            "api.endpoints.synthesis.daily_budget_exceeded", return_value=False
        ), patch(
            "api.endpoints.synthesis.acquire_synthesis_lock",
            return_value=(True, lambda: None),
        ), patch(
            "api.endpoints.synthesis.get_usage_limit_service",
            return_value=FakeUsageService(),
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service", return_value=synthesis
        ), patch(
            "api.endpoints.synthesis.record_api_calls"
        ):
            response = _post_synthesize(client, device_id=DEVICE_ID)

        assert response.status_code == 200
        synthesis.synthesize_hairstyle.assert_called_once()

    def test_reference_endpoint_rejects_anonymous(self, client, storage):
        synthesis = _synthesis_service()

        with patch(
            "api.endpoints.synthesis.daily_budget_exceeded", return_value=True
        ), patch("api.endpoints.synthesis.acquire_synthesis_lock") as mock_lock, patch(
            "api.endpoints.synthesis.get_usage_limit_service",
            return_value=FakeUsageService(),
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service", return_value=synthesis
        ):
            response = _post_reference(client, device_id=DEVICE_ID)

        assert response.status_code == 503
        assert response.json()["error"] == "daily_budget_exceeded"
        mock_lock.assert_not_called()
        synthesis.synthesize_with_reference.assert_not_called()

    def test_rejection_is_logged(self, client, storage):
        with patch(
            "api.endpoints.synthesis.daily_budget_exceeded", return_value=True
        ), patch(
            "api.endpoints.synthesis.get_usage_limit_service",
            return_value=FakeUsageService(),
        ), patch(
            "api.endpoints.synthesis.log_structured"
        ) as mock_log:
            _post_synthesize(client, device_id=DEVICE_ID)

        failures = [
            call.args[1]
            for call in mock_log.call_args_list
            if call.args[0] == "synthesis_failed"
        ]
        assert failures[-1]["reason"] == "daily_budget_exceeded"
        assert failures[-1]["status_code"] == 503
        assert failures[-1]["authenticated"] is False
        # 사전 거절은 Gemini 를 부르지 않았으므로 api_calls 필드가 없어야 한다
        assert "api_calls" not in failures[-1]

    def test_no_charge_when_rejected(self, client, storage):
        """503 으로 거절된 요청은 무료 한도도 크레딧도 소모하지 않는다"""
        usage = FakeUsageService()
        usage.check_and_increment_usage = MagicMock()

        with patch(
            "api.endpoints.synthesis.daily_budget_exceeded", return_value=True
        ), patch("api.endpoints.synthesis.get_usage_limit_service", return_value=usage):
            response = _post_synthesize(client, device_id=DEVICE_ID)

        assert response.status_code == 503
        usage.check_and_increment_usage.assert_not_called()


class TestApiCallsAreAggregated:
    def test_success_records_api_calls(self, client, storage):
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
            "api.endpoints.synthesis.record_api_calls"
        ) as mock_record:
            response = _post_synthesize(client, headers=AUTH)

        assert response.status_code == 200
        # 회원 호출도 "하루 전체" 집계에는 들어간다
        assert mock_record.call_args.args[0] == 3
        assert mock_record.call_args.kwargs["endpoint"] == "synthesize"

    def test_failed_synthesis_still_records_cost(self, client, storage):
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
            "api.endpoints.synthesis.record_api_calls"
        ) as mock_record:
            response = _post_synthesize(client, headers=AUTH)

        assert response.status_code == 422
        # 실패해도 Gemini 비용은 이미 나갔다
        assert mock_record.call_args.args[0] == 3

    def test_cache_hit_is_excluded(self, client, storage):
        storage.get_cached_result.return_value = {
            "image_base64": "aW1n",
            "image_format": "png",
        }

        with patch(
            "api.endpoints.synthesis.daily_budget_exceeded"
        ) as mock_check, patch(
            "api.endpoints.synthesis.record_api_calls"
        ) as mock_record, patch(
            "api.endpoints.synthesis.get_usage_limit_service",
            return_value=FakeUsageService(),
        ):
            response = _post_synthesize(client, device_id=DEVICE_ID)

        assert response.status_code == 200
        assert response.json()["cached"] is True
        # 캐시는 Gemini 를 부르지 않으므로 상한 확인도 집계도 하지 않는다
        mock_check.assert_not_called()
        mock_record.assert_not_called()

    def test_reference_endpoint_records_with_its_own_endpoint_name(
        self, client, storage
    ):
        credit = MagicMock()
        credit.consume.return_value = 4

        with patch(
            "api.endpoints.synthesis.acquire_synthesis_lock",
            return_value=(True, lambda: None),
        ), patch(
            "api.endpoints.synthesis.get_credit_service", return_value=credit
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service",
            return_value=_synthesis_service(api_calls=2),
        ), patch(
            "api.endpoints.synthesis.get_user_repository"
        ), patch(
            "api.endpoints.synthesis.record_api_calls"
        ) as mock_record:
            response = _post_reference(client, headers=AUTH)

        assert response.status_code == 200
        assert mock_record.call_args.args[0] == 2
        assert mock_record.call_args.kwargs["endpoint"] == "synthesize-with-reference"


class TestBudgetStoreFailureDoesNotBlock:
    def test_read_failure_lets_anonymous_through(self, client, storage, budget_limit):
        """집계를 못 읽어도 합성은 계속된다"""
        usage = FakeUsageService()
        usage.table.get_item.side_effect = Exception("DynamoDB down")
        synthesis = _synthesis_service()

        with patch(
            "core.synthesis_budget.get_usage_limit_service", return_value=usage
        ), patch(
            "api.endpoints.synthesis.get_usage_limit_service", return_value=usage
        ), patch(
            "api.endpoints.synthesis.acquire_synthesis_lock",
            return_value=(True, lambda: None),
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service", return_value=synthesis
        ), patch(
            "api.endpoints.synthesis.record_api_calls"
        ):
            response = _post_synthesize(client, device_id=DEVICE_ID)

        assert response.status_code == 200
        synthesis.synthesize_hairstyle.assert_called_once()

    def test_record_failure_does_not_break_the_response(self, client, storage):
        credit = MagicMock()
        credit.consume.return_value = 4
        usage = FakeUsageService()
        usage.table.update_item.side_effect = Exception("DynamoDB down")

        with patch(
            "core.synthesis_budget.get_usage_limit_service", return_value=usage
        ), patch(
            "api.endpoints.synthesis.acquire_synthesis_lock",
            return_value=(True, lambda: None),
        ), patch(
            "api.endpoints.synthesis.get_credit_service", return_value=credit
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service",
            return_value=_synthesis_service(api_calls=2),
        ), patch(
            "api.endpoints.synthesis.get_user_repository"
        ):
            response = _post_synthesize(client, headers=AUTH)

        assert response.status_code == 200
        assert response.json()["success"] is True


class TestEndToEndCounting:
    """집계 -> 상한 판정이 실제 테이블 목 하나로 이어지는지"""

    def test_recorded_calls_eventually_trip_the_limit(self, monkeypatch):
        monkeypatch.setattr(settings, "DAILY_SYNTHESIS_BUDGET", 5)

        total = {"count": 0}
        table = MagicMock()

        def _update(**kwargs):
            total["count"] += kwargs["ExpressionAttributeValues"][":inc"]

        table.update_item.side_effect = _update
        table.get_item.side_effect = lambda **kwargs: {
            "Item": {"count": total["count"]}
        }
        factory = lambda: _usage_service(table)

        assert daily_budget_exceeded(usage_service_factory=factory) is False

        record_api_calls(3, usage_service_factory=factory)
        assert daily_budget_exceeded(usage_service_factory=factory) is False

        # 재시도로 3회가 더 나가면 상한(5)을 넘긴다
        record_api_calls(3, usage_service_factory=factory)
        assert total["count"] == 6
        assert daily_budget_exceeded(usage_service_factory=factory) is True


class TestResponseBody:
    def test_body_shape(self):
        body = budget_exceeded_response_body()

        assert set(body) == {"success", "error", "message"}
        assert body["success"] is False
        assert body["error"] == "daily_budget_exceeded"
        assert isinstance(body["message"], str) and body["message"].strip()


# ========== (c) 예산 0(무제한)에서도 집계는 계속된다 ==========


class TestAggregationRunsAtZeroBudget:
    """상한 조회만 생략하고, 호출 기록은 값과 무관하게 항상 쓴다.

    상한을 정하려면 먼저 평소 호출량이 쌓여야 하므로 이게 깨지면
    "며칠 모아서 상한 결정" 자체가 불가능해진다.
    """

    @pytest.fixture(autouse=True)
    def _zero_budget(self, monkeypatch):
        monkeypatch.setattr(settings, "DAILY_SYNTHESIS_BUDGET", 0)

    def test_record_writes_even_when_budget_is_zero(self):
        table = MagicMock()

        record_api_calls(2, usage_service_factory=lambda: _usage_service(table))

        assert (
            table.update_item.call_args.kwargs["ExpressionAttributeValues"][":inc"] == 2
        )

    def test_check_skips_lookup_but_record_still_writes(self):
        table = MagicMock()
        factory = lambda: _usage_service(table)

        assert daily_budget_exceeded(usage_service_factory=factory) is False
        record_api_calls(1, usage_service_factory=factory)

        table.get_item.assert_not_called()  # 상한 조회는 생략
        table.update_item.assert_called_once()  # 집계는 수행

    def test_huge_recorded_total_never_blocks_at_zero(self):
        """이미 많이 썼어도 상한 0 이면 막지 않는다"""
        table = _table_with_count(1_000_000)

        assert (
            daily_budget_exceeded(usage_service_factory=lambda: _usage_service(table))
            is False
        )

    # ----- 모든 합성 경로: 성공 / 실패 / 재시도 -----

    @pytest.mark.parametrize(
        "poster,endpoint_name",
        [
            (_post_synthesize, "synthesize"),
            (_post_reference, "synthesize-with-reference"),
        ],
    )
    @pytest.mark.parametrize("success", [True, False])
    @pytest.mark.parametrize("api_calls", [1, 3])
    @pytest.mark.parametrize("anonymous", [True, False])
    def test_every_path_is_aggregated(
        self, client, storage, poster, endpoint_name, success, api_calls, anonymous
    ):
        credit = MagicMock()
        credit.consume.return_value = 4
        usage = FakeUsageService()

        with patch(
            "api.endpoints.synthesis.acquire_synthesis_lock",
            return_value=(True, lambda: None),
        ), patch(
            "api.endpoints.synthesis.get_credit_service", return_value=credit
        ), patch(
            "api.endpoints.synthesis.get_usage_limit_service", return_value=usage
        ), patch(
            "core.quota.get_usage_limit_service", return_value=usage
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service",
            return_value=_synthesis_service(success=success, api_calls=api_calls),
        ), patch(
            "api.endpoints.synthesis.get_user_repository"
        ), patch(
            "api.endpoints.synthesis.record_api_calls"
        ) as mock_record:
            response = poster(
                client,
                headers=None if anonymous else AUTH,
                device_id=DEVICE_ID if anonymous else None,
            )

        assert response.status_code == (200 if success else 422)
        # 성공이든 실패든, 회원이든 비로그인이든, 재시도가 몇 번이든 전부 기록된다
        mock_record.assert_called_once()
        assert mock_record.call_args.args[0] == api_calls
        assert mock_record.call_args.kwargs["endpoint"] == endpoint_name

    def test_zero_api_calls_is_passed_through_and_ignored_by_the_store(
        self, client, storage
    ):
        """호출 전에 실패하면 0 이 넘어오고, 저장소는 아무것도 쓰지 않는다"""
        credit = MagicMock()
        credit.consume.return_value = 4
        table = MagicMock()

        with patch(
            "api.endpoints.synthesis.acquire_synthesis_lock",
            return_value=(True, lambda: None),
        ), patch(
            "api.endpoints.synthesis.get_credit_service", return_value=credit
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service",
            return_value=_synthesis_service(success=False, api_calls=0),
        ), patch(
            "core.synthesis_budget.get_usage_limit_service",
            return_value=_usage_service(table),
        ):
            response = _post_synthesize(client, headers=AUTH)

        assert response.status_code == 422
        table.update_item.assert_not_called()

    def test_cache_hit_is_still_excluded_at_zero_budget(self, client, storage):
        storage.get_cached_result.return_value = {
            "image_base64": "aW1n",
            "image_format": "png",
        }

        with patch("api.endpoints.synthesis.record_api_calls") as mock_record, patch(
            "api.endpoints.synthesis.get_credit_service", return_value=MagicMock()
        ):
            response = _post_synthesize(client, headers=AUTH)

        assert response.status_code == 200
        assert response.json()["cached"] is True
        mock_record.assert_not_called()
