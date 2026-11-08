"""앱 서버 계약 v2.1 (docs/APP_SERVER_CONTRACT_v2.1.md) 필드 테스트

- GET /api/credits has_purchased
- 염색 합성 화질 등급 (무료 512 + 워터마크 / 유료 1K) + is_preview
- analyze-hybrid data.full_ranking
- 첫 구매 재구매 거절 422 (앱이 409 를 중복 지급으로 보고 consume 하므로)

/api/usage 필드, 실험군 429 limit_type/limit_detail, 스타일 합성 is_preview 는
test_monetization_v21.py · test_billing_flow_integration.py 에서 검증한다.
"""

import base64
import io
import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from config.settings import settings
from core.jwt_auth import create_access_token
from main import app


def _png_b64(size=(128, 128), color=(30, 30, 30)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (180, 140, 120)).save(buf, format="JPEG")
    return buf.getvalue()


def _auth(user_id: str = "u-contract") -> dict:
    return {"Authorization": f"Bearer {create_access_token(user_id)}"}


@pytest.fixture(autouse=True)
def no_rate_limit(monkeypatch):
    from api.endpoints import credits, hair_color

    for module in (credits, hair_color):
        monkeypatch.setattr(module.limiter, "enabled", False)


# ========== GET /api/credits has_purchased ==========


class TestCreditsHasPurchased:
    def _get(self, first_purchase_at, history):
        credit = MagicMock()
        credit.get_wallet.return_value = {
            "balance": 7,
            "paid_balance": 0,
            "first_purchase_at": first_purchase_at,
        }
        credit.get_history.return_value = history
        with patch(
            "api.endpoints.credits.get_credit_service", return_value=credit
        ), TestClient(app) as client:
            response = client.get("/api/credits", headers=_auth())
        assert response.status_code == 200
        return response.json()

    def test_first_purchase_at_means_purchased(self):
        # 결제가 최근 20건 밖으로 밀려나도 true
        body = self._get("2026-10-01T00:00:00+00:00", [{"reason": "synthesis"}] * 20)
        assert body["has_purchased"] is True
        assert body["balance"] == 7
        assert len(body["history"]) == 20

    def test_legacy_purchase_in_history_counts(self):
        """first_purchase_at 도입 전 결제자는 내역으로 보완"""
        body = self._get(None, [{"reason": "synthesis"}, {"reason": "purchase"}])
        assert body["has_purchased"] is True

    def test_no_purchase(self):
        body = self._get(None, [{"reason": "signup_bonus"}])
        assert body["has_purchased"] is False


# ========== 염색 합성 화질 등급 ==========

GEMINI_B64 = _png_b64()


def _hair_color_service():
    service = MagicMock()
    service.get_color_by_name.return_value = {"hex": "#C4A484"}
    service.synthesize_hair_color.side_effect = lambda **kwargs: {
        "success": True,
        "image_base64": GEMINI_B64,
        "image_format": "png",
        "message": "ok",
    }
    return service


def _hair_color_request(client, path, tier):
    quota = {"mode": "credits", "balance": 9, "tier": tier}
    service = _hair_color_service()
    # 일 예산·중복 요청 잠금은 여기의 관심사가 아니다 (test_hair_color_safeguards 가 검증)
    with patch(
        "api.endpoints.hair_color.charge_synthesis_quota",
        return_value=(None, quota, lambda: None),
    ), patch("api.endpoints.hair_color._get_service", return_value=service), patch(
        "api.endpoints.hair_color.record_api_calls"
    ), patch(
        "api.endpoints.hair_color.daily_budget_exceeded", return_value=False
    ), patch(
        "api.endpoints.hair_color.acquire_synthesis_lock",
        return_value=(True, lambda: None),
    ):
        response = client.post(
            path,
            files={"file": ("face.jpg", _jpeg_bytes(), "image/jpeg")},
            data={"color_name": "밀크브라운"},
            headers=_auth(),
        )
    return response, service


HAIR_COLOR_PATHS = ["/api/hair-color/synthesize"]


class TestHairColorTier:
    @pytest.mark.parametrize("path", HAIR_COLOR_PATHS)
    def test_free_is_512_watermarked_preview(self, path):
        with TestClient(app) as client:
            response, service = _hair_color_request(client, path, "free")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["is_preview"] is True
        assert body["quality"] == {
            "tier": "free",
            "image_size": "512",
            "watermarked": True,
        }
        assert body["image_base64"] != GEMINI_B64  # 워터마크가 들어감
        assert service.synthesize_hair_color.call_args.kwargs["image_size"] == "512"
        assert body["quota"]["tier"] == "free"  # 기존 필드 유지

    @pytest.mark.parametrize("path", HAIR_COLOR_PATHS)
    def test_paid_is_1k_without_watermark(self, path):
        with TestClient(app) as client:
            response, service = _hair_color_request(client, path, "paid")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["is_preview"] is False
        assert body["quality"]["image_size"] == "1K"
        assert body["image_base64"] == GEMINI_B64
        assert service.synthesize_hair_color.call_args.kwargs["image_size"] == "1K"

    def test_service_passes_image_size_to_gemini(self):
        from services.hair_color_service import HairColorService

        part = MagicMock()
        part.inline_data = MagicMock(
            data=base64.b64decode(_png_b64()), mime_type="image/png"
        )
        part.text = None
        response = MagicMock()
        response.candidates = [MagicMock(content=MagicMock(parts=[part]))]
        service = HairColorService()
        service._gemini_client = MagicMock()
        service._gemini_client.models.generate_content.return_value = response

        service.synthesize_hair_color(
            _jpeg_bytes(), "밀크브라운", "#C4A484", None, "512"
        )
        config = service._gemini_client.models.generate_content.call_args.kwargs[
            "config"
        ]
        # Nano Banana 는 512 미지원 → 1K 로 받고 finalize 에서 줄인다
        assert config.image_config.image_size == "1K"
        assert config.thinking_config.thinking_level.lower() == "minimal"
        assert config.response_modalities == ["IMAGE"]

        service.synthesize_hair_color(_jpeg_bytes(), "밀크브라운", "#C4A484")
        config = service._gemini_client.models.generate_content.call_args.kwargs[
            "config"
        ]
        assert config.image_config is None


# ========== analyze-hybrid full_ranking ==========


def _recommender_service(ranked, allowed=None):
    from services.hybrid_recommender import MLRecommendationService

    with patch.object(MLRecommendationService, "__init__", return_value=None):
        service = MLRecommendationService()
    service.ml_available = True
    service.reason_generator = None
    service.ml_recommender = MagicMock()
    service.ml_recommender.allowed_indices = allowed
    service.ml_recommender.recommend_top_k.side_effect = lambda **kw: ranked[: kw["k"]]
    return service


RANKED = [
    {"hairstyle_id": 1, "hairstyle": "리프컷", "score": 91.234},
    {"hairstyle_id": 2, "hairstyle": "댄디컷", "score": 88.0},
    {"hairstyle_id": 3, "hairstyle": "댄디 컷", "score": 87.0},  # 띄어쓰기 중복
    {"hairstyle_id": 4, "hairstyle": "투블럭", "score": 80.5},
    {"hairstyle_id": 5, "hairstyle": "가르마펌", "score": 70.0},
]


def _recommend(service):
    with patch(
        "services.trending_style_service.get_trending_style_service"
    ) as trending:
        trending.return_value.pick_trending.return_value = []
        return service.recommend(
            image_data=b"x",
            face_shape="계란형",
            skin_tone="봄웜",
            face_features=[1.4, 120.0, 130.0, 110.0, 0.92, 0.85],
            skin_features=[45.0, 15.0],
            gender="male",
        )


class TestFullRanking:
    def test_full_ranking_is_score_ordered_and_deduped(self):
        service = _recommender_service(RANKED)
        result = _recommend(service)

        assert result["full_ranking"] == [
            {"style_name": "리프컷", "score": 0.91},
            {"style_name": "댄디컷", "score": 0.88},
            {"style_name": "투블럭", "score": 0.81},
            {"style_name": "가르마펌", "score": 0.7},
        ]
        # Top-3 결과는 그대로 (기존 필드 유지)
        assert [r["style_name"] for r in result["recommendations"]] == [
            "리프컷",
            "댄디컷",
        ]

        first, second = service.ml_recommender.recommend_top_k.call_args_list
        assert first.kwargs["k"] == 3
        assert "log_selection" not in first.kwargs
        assert second.kwargs["k"] == settings.FULL_RANKING_LIMIT
        assert second.kwargs["log_selection"] is False
        assert second.kwargs["gender"] == "male"

    def test_limit_capped_by_allowed_candidates(self):
        """k 가 노이즈 제외 후보보다 크면 전체 후보 폴백이 일어나므로 맞춘다"""
        service = _recommender_service(RANKED, allowed=[0, 1, 2])
        _recommend(service)
        second = service.ml_recommender.recommend_top_k.call_args_list[1]
        assert second.kwargs["k"] == 3

    def test_disabled_limit_skips_second_call(self, monkeypatch):
        monkeypatch.setattr(settings, "FULL_RANKING_LIMIT", 0)
        service = _recommender_service(RANKED)
        assert _recommend(service)["full_ranking"] == []
        assert service.ml_recommender.recommend_top_k.call_count == 1

    def test_ranking_failure_keeps_recommendations(self):
        service = _recommender_service(RANKED)
        service.ml_recommender.recommend_top_k.side_effect = [
            RANKED[:3],
            RuntimeError("boom"),
        ]
        result = _recommend(service)
        assert result["full_ranking"] == []
        assert len(result["recommendations"]) == 2

    def test_analyze_hybrid_returns_full_ranking_in_data(self):
        from core.dependencies import get_face_detection_service, get_hybrid_service

        features = MagicMock(
            face_shape="계란형",
            skin_tone="봄웜",
            face_features=[1.4, 120.0, 130.0, 110.0, 0.92, 0.85],
            skin_features=[45.0, 15.0],
            confidence=0.9,
        )
        detector = MagicMock()
        detector.detect_face.return_value = {"has_face": True, "features": features}
        service = _recommender_service(RANKED)

        app.dependency_overrides[get_face_detection_service] = lambda: detector
        app.dependency_overrides[get_hybrid_service] = lambda: service
        try:
            with patch(
                "services.trending_style_service.get_trending_style_service"
            ) as trending, patch(
                "api.endpoints.analyze.save_to_database", return_value=None
            ), patch(
                "api.endpoints.analyze.limiter.enabled", False
            ), TestClient(
                app
            ) as client:
                trending.return_value.pick_trending.return_value = []
                response = client.post(
                    "/api/v2/analyze-hybrid",
                    files={"file": ("face.jpg", _jpeg_bytes(), "image/jpeg")},
                    data={"gender": "male"},
                )
        finally:
            app.dependency_overrides.pop(get_face_detection_service, None)
            app.dependency_overrides.pop(get_hybrid_service, None)

        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["full_ranking"][0] == {"style_name": "리프컷", "score": 0.91}
        assert "recommendations" in data
