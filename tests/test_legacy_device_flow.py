# -*- coding: utf-8 -*-
"""레거시 device_id 흐름 킬 스위치 + 계측 로그 테스트 (감사 항목 M4 준비)

- settings.LEGACY_DEVICE_FLOW_ENABLED=False: 비로그인 과금은 카운터를 건드리기
  전에 401, /api/usage 계열은 410
- True(기본): 기존 동작 유지 + 익명 과금 1건당 구조화 로그 1줄
  (device_id / IP 는 반드시 마스킹)
"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from config.settings import settings
from core.quota import charge_synthesis_quota, mask_device_id, mask_ip
from main import app

DEVICE_ID = "a1b2c3d4e5f60718"  # Android ID 형태 (16 hex)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def usage_service():
    """모든 카운터 호출을 기록하는 사용량 서비스 목"""
    service = MagicMock()
    service.increment_daily_counter.return_value = True
    service.check_and_increment_usage.return_value = {
        "allowed": True,
        "daily_limit": 3,
        "used": 1,
        "remaining": 2,
    }
    return service


class TestMasking:
    def test_device_id_masked_to_prefix_and_hash(self):
        masked = mask_device_id(DEVICE_ID)

        assert masked.startswith("a1b2")
        assert DEVICE_ID not in masked
        # 같은 기기는 같은 값으로 집계 가능
        assert masked == mask_device_id(DEVICE_ID)
        assert masked != mask_device_id("b1b2c3d4e5f60718")

    def test_device_id_none(self):
        assert mask_device_id(None) == "unknown"

    def test_ipv4_last_octet_zeroed(self):
        assert mask_ip("203.0.113.42") == "203.0.113.0"

    def test_ipv6_truncated(self):
        masked = mask_ip("2001:db8:1234:5678::1")

        assert masked.startswith("2001:db8:1234")
        assert "5678" not in masked

    def test_invalid_and_missing_ip(self):
        assert mask_ip(None) == "unknown"
        assert mask_ip("not-an-ip") == "invalid"


class TestKillSwitchDisabled:
    """LEGACY_DEVICE_FLOW_ENABLED=False"""

    @pytest.fixture(autouse=True)
    def disable_flag(self, monkeypatch):
        monkeypatch.setattr(settings, "LEGACY_DEVICE_FLOW_ENABLED", False)

    def test_anonymous_charge_rejected_without_touching_counters(self, usage_service):
        with pytest.raises(HTTPException) as exc_info:
            charge_synthesis_quota(
                None,
                DEVICE_ID,
                "203.0.113.42",
                endpoint="synthesize",
                usage_service_factory=lambda: usage_service,
            )

        assert exc_info.value.status_code == 401
        assert "로그인이 필요합니다" in exc_info.value.detail
        usage_service.increment_daily_counter.assert_not_called()
        usage_service.check_and_increment_usage.assert_not_called()

    def test_logged_in_user_unaffected(self, usage_service):
        credit_service = MagicMock()
        credit_service.consume.return_value = 9

        error, quota, _refund = charge_synthesis_quota(
            "user-1",
            None,
            "203.0.113.42",
            endpoint="synthesize",
            credit_service_factory=lambda: credit_service,
            usage_service_factory=lambda: usage_service,
        )

        assert error is None
        assert quota == {"mode": "credits", "balance": 9}

    def test_usage_endpoint_returns_410(self, client):
        response = client.get("/api/usage", params={"device_id": DEVICE_ID})

        assert response.status_code == 410
        assert "로그인" in response.json()["detail"]

    def test_usage_consume_endpoint_returns_410(self, client):
        response = client.post("/api/usage/consume", params={"device_id": DEVICE_ID})

        assert response.status_code == 410


class TestKillSwitchEnabled:
    """LEGACY_DEVICE_FLOW_ENABLED=True (기본) - 동작 유지 + 계측 로그"""

    @pytest.fixture(autouse=True)
    def enable_flag(self, monkeypatch):
        monkeypatch.setattr(settings, "LEGACY_DEVICE_FLOW_ENABLED", True)

    def test_anonymous_charge_emits_masked_structured_log(self, usage_service):
        with patch("core.quota.log_structured") as mock_log:
            error, quota, _refund = charge_synthesis_quota(
                None,
                DEVICE_ID,
                "203.0.113.42",
                endpoint="synthesize",
                usage_service_factory=lambda: usage_service,
            )

        assert error is None
        assert quota["mode"] == "device"

        mock_log.assert_called_once()
        event_type, data = mock_log.call_args[0]
        assert event_type == "legacy_device_flow"
        assert data["endpoint"] == "synthesize"
        assert data["ip"] == "203.0.113.0"
        assert data["device_id"].startswith("a1b2")
        # 원본 식별자는 절대 로그에 남지 않는다
        assert DEVICE_ID not in data["device_id"]
        assert "203.0.113.42" not in data.values()

    def test_logged_in_charge_does_not_log_legacy_event(self, usage_service):
        credit_service = MagicMock()
        credit_service.consume.return_value = 4

        with patch("core.quota.log_structured") as mock_log:
            charge_synthesis_quota(
                "user-1",
                DEVICE_ID,
                "203.0.113.42",
                endpoint="synthesize",
                credit_service_factory=lambda: credit_service,
                usage_service_factory=lambda: usage_service,
            )

        mock_log.assert_not_called()

    def test_usage_endpoint_still_available(self, client):
        service = MagicMock()
        service.get_usage.return_value = {"daily_limit": 3, "used": 1, "remaining": 2}

        with patch("api.endpoints.usage.get_usage_limit_service", return_value=service):
            response = client.get("/api/usage", params={"device_id": DEVICE_ID})

        assert response.status_code == 200
        assert response.json()["remaining"] == 2
