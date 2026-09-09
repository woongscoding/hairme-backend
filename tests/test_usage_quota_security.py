# -*- coding: utf-8 -*-
"""합성 과금/무료 한도 우회 차단 테스트

- hair-color 합성도 로그인 시 크레딧 과금 (+ 실패 시 환불)
- 비로그인 합성은 device_id 로테이션으로 우회 불가 (IP 일일 상한)
- device_id 형식 검증 ('#' 네임스페이스 주입 차단)
- 입력 검증 실패 시 사용량이 차감되지 않음
"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from config.settings import settings
from core.jwt_auth import create_access_token
from main import app
from services.usage_limit_service import validate_device_id


DEVICE_ID = "a1b2c3d4e5f60718"  # Android ID 형태 (16 hex)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 90, 60)).save(buf, "PNG")
    return buf.getvalue()


class FakeUsageService:
    """DynamoDB 없이 동작하는 사용량 카운터 (동작 검증용)"""

    def __init__(self):
        self.counters = {}
        self.device_calls = []

    def increment_daily_counter(self, key: str, limit: int) -> bool:
        current = self.counters.get(key, 0)
        if current >= limit:
            return False
        self.counters[key] = current + 1
        return True

    def decrement_daily_counter(self, key: str) -> None:
        self.counters[key] = max(0, self.counters.get(key, 0) - 1)

    def check_and_increment_usage(self, device_id: str):
        self.device_calls.append(device_id)
        limit = settings.DAILY_SYNTHESIS_LIMIT
        current = self.counters.get(device_id, 0)
        if current >= limit:
            return {
                "allowed": False,
                "daily_limit": limit,
                "used": current,
                "remaining": 0,
            }
        self.counters[device_id] = current + 1
        return {
            "allowed": True,
            "daily_limit": limit,
            "used": current + 1,
            "remaining": limit - (current + 1),
        }

    def decrement_usage(self, device_id: str) -> None:
        self.decrement_daily_counter(device_id)


def _hair_color_service(success: bool = True) -> MagicMock:
    service = MagicMock()
    service.get_color_by_name.return_value = {"hex": "#8B4513"}
    if success:
        service.synthesize_hair_color.return_value = {
            "success": True,
            "image_base64": "aW1n",
            "image_format": "png",
            "message": "적용되었습니다.",
        }
    else:
        service.synthesize_hair_color.return_value = {
            "success": False,
            "message": "얼굴을 찾지 못했습니다",
        }
    return service


def _post_hair_color(client, *, device_id=None, headers=None):
    data = {"color_name": "밀크브라운"}
    if device_id:
        data["device_id"] = device_id
    return client.post(
        "/api/hair-color/synthesize",
        files={"file": ("face.png", _png_bytes(), "image/png")},
        data=data,
        headers=headers or {},
    )


# ========== (a) 로그인 회원: hair-color 합성도 크레딧 과금 ==========


class TestHairColorCreditBilling:
    AUTH = {"Authorization": f"Bearer {create_access_token('color-user')}"}

    def test_logged_in_user_consumes_credit(self, client):
        credit = MagicMock()
        credit.consume.return_value = 4
        usage = FakeUsageService()

        with patch("core.quota.get_credit_service", return_value=credit), patch(
            "core.quota.get_usage_limit_service", return_value=usage
        ), patch(
            "api.endpoints.hair_color._get_service",
            return_value=_hair_color_service(True),
        ):
            response = _post_hair_color(client, headers=self.AUTH)

        assert response.status_code == 200
        credit.consume.assert_called_once()
        credit.grant.assert_not_called()
        # 회원은 device 무료 한도를 소비하지 않는다
        assert usage.device_calls == []
        assert response.json()["quota"] == {"mode": "credits", "balance": 4}

    def test_failed_synthesis_refunds_credit(self, client):
        credit = MagicMock()
        credit.consume.return_value = 4

        with patch("core.quota.get_credit_service", return_value=credit), patch(
            "core.quota.get_usage_limit_service", return_value=FakeUsageService()
        ), patch(
            "api.endpoints.hair_color._get_service",
            return_value=_hair_color_service(False),
        ):
            response = _post_hair_color(client, headers=self.AUTH)

        assert response.status_code == 422
        assert credit.grant.call_args.kwargs.get("reason") == "refund"


# ========== (b) 비로그인: device_id 로테이션으로도 IP 상한을 넘길 수 없음 ==========


class TestAnonymousIpCap:
    def test_rotating_device_ids_hit_ip_cap(self, client, monkeypatch):
        monkeypatch.setattr(settings, "ANON_IP_DAILY_SYNTHESIS_LIMIT", 1)
        usage = FakeUsageService()

        with patch("core.quota.get_usage_limit_service", return_value=usage), patch(
            "api.endpoints.hair_color._get_service",
            return_value=_hair_color_service(True),
        ):
            first = _post_hair_color(client, device_id="aaaa1111bbbb2222")
            # 매번 새 device_id를 만들어도 같은 IP면 막혀야 한다
            second = _post_hair_color(client, device_id="cccc3333dddd4444")

        assert first.status_code == 200
        assert second.status_code == 429
        body = second.json()
        assert body["error"] == "daily_limit_exceeded"
        assert body["limit_type"] == "ip"
        # IP 상한에 걸린 요청은 device 카운터를 소비하지 않는다
        assert usage.device_calls == ["aaaa1111bbbb2222"]


# ========== (c) device_id 형식 검증 ('#' 네임스페이스 주입 차단) ==========


class TestDeviceIdValidation:
    def test_validate_device_id_accepts_common_formats(self):
        assert validate_device_id(" a1b2c3d4e5f60718 ") == "a1b2c3d4e5f60718"
        uuid_like = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
        assert validate_device_id(uuid_like) == uuid_like

    @pytest.mark.parametrize(
        "bad",
        [
            "reward_ad#victim-user",
            "ip#127.0.0.1",
            "short",
            "has space here",
            "",
        ],
    )
    def test_validate_device_id_rejects(self, bad):
        with pytest.raises(ValueError):
            validate_device_id(bad)

    def test_usage_endpoint_rejects_namespaced_device_id(self, client):
        response = client.get("/api/usage", params={"device_id": "reward_ad#victim"})
        assert response.status_code == 400

    def test_usage_consume_endpoint_rejects_namespaced_device_id(self, client):
        response = client.post(
            "/api/usage/consume", params={"device_id": "reward_ad#victim"}
        )
        assert response.status_code == 400

    def test_synthesis_rejects_namespaced_device_id(self, client):
        storage = MagicMock()
        storage.enabled = False
        storage.build_cache_key.return_value = "cache-key"
        storage.get_cached_result.return_value = None
        usage = FakeUsageService()

        with patch(
            "api.endpoints.synthesis.get_photo_storage_service", return_value=storage
        ), patch("core.quota.get_usage_limit_service", return_value=usage):
            response = client.post(
                "/api/v2/synthesize",
                files={"file": ("face.png", _png_bytes(), "image/png")},
                data={
                    "hairstyle_name": "크롭컷",
                    "gender": "male",
                    "device_id": "reward_ad#victim-user",
                },
            )

        assert response.status_code == 400
        # 타인의 리워드 카운터가 조작되어서는 안 된다
        assert usage.counters == {}


# ========== (d) 입력 검증 실패 시 사용량 차감 없음 ==========


class TestValidationBeforeCharge:
    def test_invalid_color_index_does_not_consume_usage(self, client):
        usage = FakeUsageService()
        service = _hair_color_service(True)
        recommendations = MagicMock()
        recommendations.recommended = [MagicMock()]
        service.get_recommendations.return_value = recommendations

        with patch("core.quota.get_usage_limit_service", return_value=usage), patch(
            "api.endpoints.hair_color._get_service", return_value=service
        ):
            response = client.post(
                "/api/hair-color/synthesize-by-personal-color",
                files={"file": ("face.png", _png_bytes(), "image/png")},
                data={
                    "personal_color": "봄웜",
                    "color_index": 99,
                    "device_id": DEVICE_ID,
                },
            )

        assert response.status_code == 400
        assert usage.counters == {}
        assert usage.device_calls == []
        service.synthesize_hair_color.assert_not_called()
