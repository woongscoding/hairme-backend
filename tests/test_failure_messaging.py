# -*- coding: utf-8 -*-
"""실패·거절 안내와 대가성 문구 테스트 (T-004)

앱이 JSON 원문을 그대로 보여주던 문제를 서버 쪽에서 먼저 정리한다.

- 합성 성공 4경로(스타일/레퍼런스 × 캐시히트/신규) 응답에 disclosure 포함
  (추천 제품이 함께 내려가므로 공정위 표시광고 문구가 빠지면 안 된다)
- 합성 거절 422 본문에 error 코드 / refunded / balance
- 중복 요청 409 는 "다시 시도" 를 안내한다 (받을 결과가 있다는 오해 제거)
- 비로그인 429 는 로그인을, 회원 402 는 잔액이 빈 이유(가입 보너스 1회성)를 알린다
  (충전/광고 시청 같은 액션 문구는 서버가 넣지 않는다)
- /v2/analyze-hybrid 의 400 은 analysis_error 로 원인(얼굴 미감지/포맷)을 구분해 센다

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
from main import app

DEVICE_ID = "a1b2c3d4e5f60718"
USER_ID = "failure-messaging-user"
CACHE_KEY = "e" * 64
AUTH = {"Authorization": f"Bearer {create_access_token(USER_ID)}"}


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _no_rate_limit():
    """합성 엔드포인트의 분당 제한은 이 테스트의 관심사가 아니다"""
    from api.endpoints import synthesis as synthesis_module

    previous = synthesis_module.limiter.enabled
    synthesis_module.limiter.enabled = False
    yield
    synthesis_module.limiter.enabled = previous


@pytest.fixture(autouse=True)
def _no_budget_writes():
    """일 예산 집계/상한은 여기의 관심사가 아니다 (실제 DynamoDB 접근 차단)"""
    with patch("api.endpoints.synthesis.record_api_calls"), patch(
        "api.endpoints.synthesis.daily_budget_exceeded", return_value=False
    ):
        yield


@pytest.fixture(autouse=True)
def _no_lock_writes():
    """중복 요청 잠금은 통과시킨다 (잠금 자체는 test_synthesis_cost_guard 가 검증)"""
    with patch(
        "api.endpoints.synthesis.acquire_synthesis_lock",
        return_value=(True, lambda: None),
    ):
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


def _synthesis_service(success=True):
    service = MagicMock()
    service.synthesize_hairstyle.return_value = {
        "success": success,
        "image_base64": "aW1n" if success else None,
        "image_format": "png" if success else None,
        "message": (
            "적용되었습니다." if success else "AI가 잠깐 헤맸어요.. 다시 시도해주세요!"
        ),
        "api_calls": 1,
    }
    service.synthesize_with_reference.return_value = (
        service.synthesize_hairstyle.return_value
    )
    return service


class FakeUsageService:
    """DynamoDB 없이 동작하는 사용량 카운터 (한도 소진 상태를 만들 수 있다)"""

    def __init__(self, device_allowed=True):
        self.device_allowed = device_allowed
        self.decremented = []

    def increment_daily_counter(self, key, limit):
        return True

    def decrement_daily_counter(self, key):
        self.decremented.append(key)

    def check_and_increment_usage(self, device_id):
        limit = settings.DAILY_SYNTHESIS_LIMIT
        if not self.device_allowed:
            return {
                "allowed": False,
                "daily_limit": limit,
                "used": limit,
                "remaining": 0,
            }
        return {
            "allowed": True,
            "daily_limit": limit,
            "used": 1,
            "remaining": limit - 1,
        }

    def decrement_usage(self, device_id):
        self.decremented.append(device_id)


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


def _member_credit(balance=4):
    credit = MagicMock()
    credit.consume.return_value = balance
    return credit


# ========== (1) 합성 성공 4경로의 대가성 문구 ==========


class TestDisclosureOnEverySuccessPath:
    """recommended_products 를 내려주는 응답에는 disclosure 가 항상 붙는다"""

    def _assert_disclosure(self, response):
        assert response.status_code == 200
        body = response.json()
        assert body["recommended_products"]  # 문구가 필요한 상황 자체를 확인
        assert "쿠팡 파트너스" in body["disclosure"]

    def test_style_synthesis(self, client, storage):
        with patch(
            "api.endpoints.synthesis.get_credit_service", return_value=_member_credit()
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service",
            return_value=_synthesis_service(),
        ), patch(
            "api.endpoints.synthesis.get_user_repository"
        ):
            response = _post_synthesize(client, headers=AUTH)

        self._assert_disclosure(response)

    def test_style_synthesis_cache_hit(self, client, storage):
        storage.get_cached_result.return_value = {
            "image_base64": "aW1n",
            "image_format": "png",
        }

        with patch(
            "api.endpoints.synthesis.get_credit_service", return_value=MagicMock()
        ):
            response = _post_synthesize(client, headers=AUTH)

        assert response.json()["cached"] is True
        self._assert_disclosure(response)

    def test_reference_synthesis(self, client, storage):
        with patch(
            "api.endpoints.synthesis.get_credit_service", return_value=_member_credit()
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service",
            return_value=_synthesis_service(),
        ), patch(
            "api.endpoints.synthesis.get_user_repository"
        ):
            response = _post_reference(client, headers=AUTH)

        self._assert_disclosure(response)

    def test_reference_synthesis_cache_hit(self, client, storage):
        storage.get_cached_result.return_value = {
            "image_base64": "aW1n",
            "image_format": "png",
        }

        with patch(
            "api.endpoints.synthesis.get_credit_service", return_value=MagicMock()
        ):
            response = _post_reference(client, headers=AUTH)

        assert response.json()["cached"] is True
        self._assert_disclosure(response)

    def test_disclosure_lookup_never_breaks_the_response(self, client, storage):
        """카탈로그가 깨져도 합성 응답은 살아 있어야 한다"""
        from api.endpoints.synthesis import _safe_disclosure

        with patch(
            "api.endpoints.synthesis.get_product_recommendation_service",
            side_effect=RuntimeError("catalog broken"),
        ):
            assert _safe_disclosure() == ""


# ========== (2) 합성 거절 422 본문 ==========


class TestRejected422Body:
    def test_member_body_has_error_refunded_and_restored_balance(self, client, storage):
        credit = _member_credit(balance=4)

        with patch(
            "api.endpoints.synthesis.get_credit_service", return_value=credit
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service",
            return_value=_synthesis_service(success=False),
        ):
            response = _post_synthesize(client, headers=AUTH)

        assert response.status_code == 422
        body = response.json()
        assert body["success"] is False
        assert body["error"] == "synthesis_rejected"
        assert body["refunded"] is True
        # 차감 직후 4 -> 환불로 되돌아온 잔액
        assert body["balance"] == 4 + settings.SYNTHESIS_CREDIT_COST
        assert "다시 시도" in body["message"]
        credit.grant.assert_called_once()

    def test_reference_endpoint_body_matches(self, client, storage):
        with patch(
            "api.endpoints.synthesis.get_credit_service", return_value=_member_credit()
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service",
            return_value=_synthesis_service(success=False),
        ):
            response = _post_reference(client, headers=AUTH)

        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "synthesis_rejected"
        assert body["refunded"] is True
        assert body["balance"] == 4 + settings.SYNTHESIS_CREDIT_COST

    def test_anonymous_body_has_null_balance(self, client, storage):
        """비로그인은 크레딧 잔액 개념이 없으므로 balance 는 null (필드는 존재)"""
        usage = FakeUsageService()

        with patch(
            "api.endpoints.synthesis.get_usage_limit_service", return_value=usage
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service",
            return_value=_synthesis_service(success=False),
        ):
            response = _post_synthesize(client, device_id=DEVICE_ID)

        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "synthesis_rejected"
        assert body["refunded"] is True
        assert body["balance"] is None
        # 무료 한도는 실제로 복구된다
        assert DEVICE_ID in usage.decremented

    def test_hair_color_body_matches(self, client):
        """염색 합성도 같은 실패 스키마를 쓴다 (앱이 한 가지 규약만 읽으면 된다)"""
        credit = _member_credit()
        color_service = MagicMock()
        color_service.get_color_by_name.return_value = {"hex": "#8B4513"}
        color_service.synthesize_hair_color.return_value = {
            "success": False,
            "message": "AI가 잠깐 헤맸어요.. 다시 시도해주세요!",
        }

        with patch("core.quota.get_credit_service", return_value=credit), patch(
            "api.endpoints.hair_color._get_service", return_value=color_service
        ):
            response = client.post(
                "/api/hair-color/synthesize",
                files={"file": ("face.png", _png_bytes(), "image/png")},
                data={"color_name": "밀크브라운"},
                headers=AUTH,
            )

        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "synthesis_rejected"
        assert body["refunded"] is True
        assert body["balance"] == 4 + settings.SYNTHESIS_CREDIT_COST
        credit.grant.assert_called_once()


# ========== (3) 중복 요청 409 문구 ==========


class TestDuplicateRequestMessage:
    def test_message_tells_the_user_to_retry(self):
        from core.synthesis_lock import duplicate_request_response_body

        body = duplicate_request_response_body()

        assert body["error"] == "synthesis_in_progress"
        assert "다시 시도" in body["message"]
        # 받을 결과가 어딘가 있다는 오해를 주지 않는다
        assert "결과를 확인" not in body["message"]


# ========== (4) 한도 거절 문구 (충전·광고 액션은 넣지 않는다) ==========


class TestQuotaDeniedMessages:
    def test_device_limit_invites_login(self, client, storage):
        usage = FakeUsageService(device_allowed=False)

        with patch(
            "api.endpoints.synthesis.get_usage_limit_service", return_value=usage
        ), patch("api.endpoints.synthesis.get_synthesis_service") as synthesis:
            response = _post_synthesize(client, device_id=DEVICE_ID)

        assert response.status_code == 429
        body = response.json()
        assert body["error"] == "daily_limit_exceeded"
        assert body["limit_type"] == "device"
        assert "로그인" in body["message"]
        assert str(settings.SIGNUP_BONUS_CREDITS) in body["message"]
        synthesis.assert_not_called()

    def test_insufficient_credits_explains_one_time_bonus(self, client, storage):
        from services.credit_service import InsufficientCreditsError

        credit = MagicMock()
        credit.consume.side_effect = InsufficientCreditsError(0)

        with patch("api.endpoints.synthesis.get_credit_service", return_value=credit):
            response = _post_synthesize(client, headers=AUTH)

        assert response.status_code == 402
        body = response.json()
        assert body["error"] == "insufficient_credits"
        assert body["balance"] == 0
        message = body["message"]
        assert "가입 보너스" in message
        assert str(settings.SIGNUP_BONUS_CREDITS) in message
        # 충전/광고 유도는 서버 문구가 아니라 앱이 결정한다
        assert "충전" not in message
        assert "광고" not in message


# ========== (5) /v2 400 경로의 analysis_error 계측 ==========


@pytest.fixture
def analyze_client():
    """얼굴 미검출을 반환하는 검출 서비스로 대체한 클라이언트"""
    from core.dependencies import get_face_detection_service, get_hybrid_service

    face_detector = MagicMock()
    face_detector.detect_face.return_value = {
        "has_face": False,
        "face_count": 0,
        "method": "mediapipe",
        "features": None,
    }

    app.dependency_overrides[get_face_detection_service] = lambda: face_detector
    app.dependency_overrides[get_hybrid_service] = lambda: MagicMock()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.pop(get_face_detection_service, None)
    app.dependency_overrides.pop(get_hybrid_service, None)


def _events(mock_log, event_type):
    return [
        call.args[1] for call in mock_log.call_args_list if call.args[0] == event_type
    ]


def _post_v2(client, filename="face.png"):
    return client.post(
        "/api/v2/analyze-hybrid",
        files={"file": (filename, _png_bytes(), "image/png")},
        data={"gender": "male"},
    )


class TestV2AnalysisErrorInstrumentation:
    def test_no_face_is_counted(self, analyze_client):
        with patch("api.endpoints.analyze.log_structured") as mock_log:
            response = _post_v2(analyze_client)

        assert response.status_code == 400
        events = _events(mock_log, "analysis_error")
        assert len(events) == 1
        assert events[0]["endpoint"] == "v2/analyze-hybrid"
        assert events[0]["error_type"] == "no_face_detected"
        assert events[0]["status_code"] == 400
        assert events[0]["authenticated"] is False

    def test_bad_format_is_counted_separately(self, analyze_client):
        """포맷 오류는 얼굴 미감지와 구분된다 (image_hash 계산 전에 걸린다)"""
        with patch("api.endpoints.analyze.log_structured") as mock_log:
            response = _post_v2(analyze_client, filename="face.txt")

        assert response.status_code == 400
        events = _events(mock_log, "analysis_error")
        assert len(events) == 1
        assert events[0]["error_type"] == "invalid_file_format"
        assert events[0]["image_hash"] == "unknown"
