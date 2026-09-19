# -*- coding: utf-8 -*-
"""최소 계측 테스트 (작업 1)

품질 수정 전후를 비교할 기록이 배포 전에 남는지 확인한다.

- v2 분석 경로(/api/v2/analyze-hybrid)의 analysis_complete 구조화 로그
- hairme-analysis 저장 항목의 model_version(항상) / gender(있을 때)
- 피드백 선택 필드 dislike_reason (없어도 기존 요청은 그대로 통과)
- GET /api/me/results 의 results_viewed 로그
- 어느 로그에도 user_id / device_id 원문과 사진이 남지 않는다
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
from core.logging import mask_user_id
from main import app

USER_ID = "instrumentation-user-id"
ANALYSIS_ID = "11111111-2222-3333-4444-555555555555"


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def _recommendation_result():
    """meta 에 원점수 통계가 담긴 추천 결과 (hybrid_recommender 출력 형태)"""
    return {
        "analysis": {
            "face_shape": "계란형",
            "personal_color": "봄웜",
            "features": "ML 모델 기반 분석",
        },
        "recommendations": [
            {
                "hairstyle_id": 1,
                "style_name": "레이어드 컷",
                "reason": "잘 어울립니다",
                "source": "ml",
                "score": 0.52,
                "rank": 1,
            },
            {
                "hairstyle_id": 2,
                "style_name": "시스루뱅",
                "reason": "잘 어울립니다",
                "source": "ml",
                "score": 0.49,
                "rank": 2,
            },
        ],
        "meta": {
            "total_count": 2,
            "ml_count": 2,
            "trending_count": 0,
            "method": "ml",
            "model_version": "v6_feedback_20260909",
            "top_score": 52.31,
            "score_stddev": 1.84,
        },
    }


@pytest.fixture
def analyze_client():
    """얼굴 검출 + ML 추천을 DI 오버라이드로 대체한 클라이언트"""
    from core.dependencies import get_face_detection_service, get_hybrid_service

    mp_features = MagicMock()
    mp_features.face_shape = "계란형"
    mp_features.skin_tone = "봄웜"
    mp_features.confidence = 0.9
    mp_features.face_features = None
    mp_features.skin_features = None

    face_detector = MagicMock()
    face_detector.detect_face.return_value = {
        "has_face": True,
        "face_count": 1,
        "method": "mediapipe",
        "features": mp_features,
    }

    recommender = MagicMock()
    recommender.recommend.return_value = _recommendation_result()

    app.dependency_overrides[get_face_detection_service] = lambda: face_detector
    app.dependency_overrides[get_hybrid_service] = lambda: recommender

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.pop(get_face_detection_service, None)
    app.dependency_overrides.pop(get_hybrid_service, None)


def _post_v2(client, headers=None):
    return client.post(
        "/api/v2/analyze-hybrid",
        files={"file": ("face.png", _png_bytes(), "image/png")},
        data={"gender": "male"},
        headers=headers or {},
    )


def _events(mock_log, event_type):
    return [
        call.args[1] for call in mock_log.call_args_list if call.args[0] == event_type
    ]


# ========== (1) v2 analysis_complete 로그 ==========


class TestV2AnalysisCompleteLog:
    def test_v2_emits_analysis_complete_with_quality_fields(self, analyze_client):
        with patch(
            "api.endpoints.analyze.save_to_database", return_value=ANALYSIS_ID
        ), patch("api.endpoints.analyze.log_structured") as mock_log:
            response = _post_v2(analyze_client)

        assert response.status_code == 200

        events = _events(mock_log, "analysis_complete")
        assert len(events) == 1
        data = events[0]

        assert data["endpoint"] == "v2/analyze-hybrid"
        assert data["model_version"] == "v6_feedback_20260909"
        assert data["face_shape"] == "계란형"
        assert data["top_style"] == "레이어드 컷"
        assert data["top_score"] == 52.31
        assert data["score_stddev"] == 1.84
        assert data["authenticated"] is False
        assert isinstance(data["processing_time"], float)
        assert data["analysis_id"] == ANALYSIS_ID

    def test_authenticated_flag_set_for_bearer_token(self, analyze_client):
        headers = {"Authorization": f"Bearer {create_access_token(USER_ID)}"}

        with patch(
            "api.endpoints.analyze.save_to_database", return_value=ANALYSIS_ID
        ), patch("api.endpoints.analyze.log_structured") as mock_log:
            response = _post_v2(analyze_client, headers=headers)

        assert response.status_code == 200
        data = _events(mock_log, "analysis_complete")[0]
        assert data["authenticated"] is True
        # 회원이어도 user_id 원문은 로그에 남지 않는다
        assert USER_ID not in str(data)

    def test_missing_meta_falls_back_to_unknown_model_version(self, analyze_client):
        """meta 가 없는 추천 결과(구버전/폴백)에서도 로그가 깨지지 않는다"""
        from core.dependencies import get_hybrid_service

        result = _recommendation_result()
        del result["meta"]
        app.dependency_overrides[get_hybrid_service]().recommend.return_value = result

        with patch(
            "api.endpoints.analyze.save_to_database", return_value=ANALYSIS_ID
        ), patch("api.endpoints.analyze.log_structured") as mock_log:
            response = _post_v2(analyze_client)

        assert response.status_code == 200
        data = _events(mock_log, "analysis_complete")[0]
        assert data["model_version"] == "unknown"
        assert data["top_score"] is None
        assert data["score_stddev"] is None

    def test_model_version_and_gender_passed_to_storage(self, analyze_client):
        with patch(
            "api.endpoints.analyze.save_to_database", return_value=ANALYSIS_ID
        ) as mock_save, patch("api.endpoints.analyze.log_structured"):
            response = _post_v2(analyze_client)

        assert response.status_code == 200
        kwargs = mock_save.call_args.kwargs
        assert kwargs["model_version"] == "v6_feedback_20260909"
        assert kwargs["gender"] == "male"

    def test_response_shape_unchanged(self, analyze_client):
        """구버전 앱이 읽는 기존 응답 필드는 그대로 유지된다"""
        with patch(
            "api.endpoints.analyze.save_to_database", return_value=ANALYSIS_ID
        ), patch("api.endpoints.analyze.log_structured"):
            body = _post_v2(analyze_client).json()

        for key in (
            "success",
            "analysis_id",
            "data",
            "processing_time",
            "method",
            "mediapipe_features",
            "model_used",
            "feedback_enabled",
        ):
            assert key in body
        assert body["data"]["recommendations"][0]["style_name"] == "레이어드 컷"


# ========== (2) 추천기 meta 의 원점수 통계 ==========


class TestScoreStats:
    def test_stats_use_raw_scores_not_rounded_response_scores(self):
        from services.hybrid_recommender import MLRecommendationService

        stats = MLRecommendationService._score_stats(
            [{"score": 52.31}, {"score": 49.12}, {"score": 47.80}]
        )

        assert stats["top_score"] == 52.31
        # 응답 score(0-1, 소수 2자리)로는 표현되지 않는 해상도가 남아야 한다
        assert stats["score_stddev"] > 0
        assert round(stats["score_stddev"], 3) == stats["score_stddev"]

    def test_empty_recommendations(self):
        from services.hybrid_recommender import MLRecommendationService

        assert MLRecommendationService._score_stats([]) == {
            "top_score": None,
            "score_stddev": None,
        }


# ========== (3) DynamoDB 저장: model_version 항상, gender 선택 ==========


class TestAnalysisPersistence:
    def _save(self, data):
        import database.dynamodb_connection as db

        table = MagicMock()
        with patch.object(db, "dynamodb_enabled", True), patch.object(
            db, "dynamodb_table", table
        ):
            analysis_id = db.save_analysis(data)
        return analysis_id, table.put_item.call_args.kwargs["Item"]

    def test_model_version_always_present(self):
        _id, item = self._save(
            {
                "face_shape": "계란형",
                "personal_color": "봄웜",
                "recommendations": [],
                "processing_time": 1.0,
                "detection_method": "ml",
                "image_hash": "abc",
            }
        )

        assert item["model_version"] == "unknown"

    def test_model_version_and_gender_stored(self):
        _id, item = self._save(
            {
                "face_shape": "계란형",
                "personal_color": "봄웜",
                "recommendations": [],
                "processing_time": 1.0,
                "detection_method": "ml",
                "image_hash": "abc",
                "model_version": "v6_feedback_20260909",
                "gender": "female",
            }
        )

        assert item["model_version"] == "v6_feedback_20260909"
        assert item["gender"] == "female"

    def test_gender_omitted_when_absent(self):
        _id, item = self._save(
            {
                "face_shape": "계란형",
                "personal_color": "봄웜",
                "recommendations": [],
                "processing_time": 1.0,
                "detection_method": "ml",
                "image_hash": "abc",
            }
        )

        assert "gender" not in item

    def test_repository_forwards_model_version_and_gender(self):
        from database.dynamodb_repository import DynamoDBAnalysisRepository

        with patch(
            "database.dynamodb_repository.save_analysis", return_value=ANALYSIS_ID
        ) as mock_save:
            DynamoDBAnalysisRepository().save_analysis(
                image_hash="abc",
                analysis_result={
                    "analysis": {"face_shape": "계란형", "personal_color": "봄웜"},
                    "recommendations": [],
                },
                processing_time=1.0,
                detection_method="ml",
                model_version="v6_feedback_20260909",
                gender="male",
            )

        data = mock_save.call_args.args[0]
        assert data["model_version"] == "v6_feedback_20260909"
        assert data["gender"] == "male"


# ========== (4) 피드백 dislike_reason ==========


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestFeedbackDislikeReason:
    def _post(self, client, payload):
        with patch(
            "api.endpoints.feedback.get_analysis",
            return_value={"analysis_id": ANALYSIS_ID},
        ), patch(
            "api.endpoints.feedback.save_feedback", return_value=True
        ) as mock_save, patch(
            "api.endpoints.feedback.log_structured"
        ) as mock_log:
            response = client.post("/api/feedback", json=payload)
        return response, mock_save, mock_log

    def test_accepted_and_stored_and_logged(self, client):
        response, mock_save, mock_log = self._post(
            client,
            {
                "analysis_id": ANALYSIS_ID,
                "style_index": 1,
                "feedback": "dislike",
                "dislike_reason": "hair_unnatural",
            },
        )

        assert response.status_code == 200
        assert mock_save.call_args.kwargs["dislike_reason"] == "hair_unnatural"
        assert mock_log.call_args.args[1]["dislike_reason"] == "hair_unnatural"

    @pytest.mark.parametrize(
        "reason", ["face_changed", "hair_unnatural", "not_my_taste"]
    )
    def test_all_allowed_values(self, client, reason):
        response, mock_save, _log = self._post(
            client,
            {
                "analysis_id": ANALYSIS_ID,
                "style_index": 2,
                "feedback": "bad",
                "dislike_reason": reason,
            },
        )

        assert response.status_code == 200
        assert mock_save.call_args.kwargs["dislike_reason"] == reason

    def test_legacy_request_without_field_still_passes(self, client):
        """구버전 앱 요청(필드 없음)은 그대로 통과하고 None 으로 저장된다"""
        response, mock_save, mock_log = self._post(
            client,
            {
                "analysis_id": ANALYSIS_ID,
                "style_index": 1,
                "feedback": "good",
                "naver_clicked": False,
            },
        )

        assert response.status_code == 200
        assert response.json()["success"] is True
        assert mock_save.call_args.kwargs["dislike_reason"] is None
        assert mock_log.call_args.args[1]["dislike_reason"] is None

    def test_unknown_value_rejected(self, client):
        response, _save, _log = self._post(
            client,
            {
                "analysis_id": ANALYSIS_ID,
                "style_index": 1,
                "feedback": "dislike",
                "dislike_reason": "too_expensive",
            },
        )

        assert response.status_code == 422

    def test_response_schema_unchanged(self, client):
        response, _save, _log = self._post(
            client,
            {
                "analysis_id": ANALYSIS_ID,
                "style_index": 3,
                "feedback": "dislike",
                "dislike_reason": "face_changed",
            },
        )

        body = response.json()
        assert set(body) == {"success", "message", "analysis_id", "style_index"}


class TestFeedbackPersistence:
    def _save_feedback(self, **kwargs):
        import database.dynamodb_connection as db

        table = MagicMock()
        with patch.object(db, "dynamodb_enabled", True), patch.object(
            db, "dynamodb_table", table
        ), patch.object(db, "redis_client", None), patch.object(
            db, "_save_feedback_to_s3"
        ):
            ok = db.save_feedback(**kwargs)
        return ok, table.update_item.call_args.kwargs

    def test_dislike_reason_written_to_item(self):
        ok, kwargs = self._save_feedback(
            analysis_id=ANALYSIS_ID,
            style_index=2,
            feedback="bad",
            naver_clicked=False,
            dislike_reason="not_my_taste",
        )

        assert ok is True
        assert "style_2_dislike_reason = :dislike_reason" in kwargs["UpdateExpression"]
        assert kwargs["ExpressionAttributeValues"][":dislike_reason"] == "not_my_taste"

    def test_absent_reason_does_not_touch_the_attribute(self):
        ok, kwargs = self._save_feedback(
            analysis_id=ANALYSIS_ID,
            style_index=2,
            feedback="good",
            naver_clicked=True,
        )

        assert ok is True
        assert "dislike_reason" not in kwargs["UpdateExpression"]
        assert ":dislike_reason" not in kwargs["ExpressionAttributeValues"]
        # 기존 필드는 그대로
        assert "style_2_feedback = :feedback" in kwargs["UpdateExpression"]
        assert "style_2_naver_clicked = :clicked" in kwargs["UpdateExpression"]


# ========== (5) results_viewed 로그 ==========


class TestResultsViewedLog:
    @pytest.fixture
    def storage(self):
        service = MagicMock()
        service.list_user_results.return_value = {
            "items": [{"key": "results/x/1.png", "url": "https://s3/1"}],
            "next_token": None,
        }
        with patch(
            "api.endpoints.results.get_photo_storage_service", return_value=service
        ):
            yield service

    def test_logs_masked_user_id(self, client, storage):
        headers = {"Authorization": f"Bearer {create_access_token(USER_ID)}"}

        with patch("api.endpoints.results.log_structured") as mock_log:
            response = client.get("/api/me/results", headers=headers)

        assert response.status_code == 200
        event_type, data = mock_log.call_args.args
        assert event_type == "results_viewed"
        assert data["count"] == 1
        assert data["has_next"] is False
        assert data["paged"] is False
        # 원문 user_id 는 남지 않는다
        assert data["user_id"] == mask_user_id(USER_ID)
        assert USER_ID not in str(data)

    def test_no_log_when_lookup_fails(self, client, storage):
        storage.list_user_results.side_effect = Exception("S3 down")
        headers = {"Authorization": f"Bearer {create_access_token(USER_ID)}"}

        with patch("api.endpoints.results.log_structured") as mock_log:
            response = client.get("/api/me/results", headers=headers)

        assert response.status_code == 500
        mock_log.assert_not_called()


# ========== (6) 마스킹 규칙 ==========


class TestMaskUserId:
    def test_prefix_and_hash_are_stable(self):
        masked = mask_user_id(USER_ID)

        assert masked.startswith(USER_ID[:4])
        assert USER_ID not in masked
        assert masked == mask_user_id(USER_ID)
        assert masked != mask_user_id("other-user-id")

    def test_missing_value(self):
        assert mask_user_id(None) == "unknown"
        assert mask_user_id("") == "unknown"

    def test_matches_existing_device_id_masking(self):
        """커밋 9d8407a 의 마스킹 방식과 동일해야 한다"""
        from core.quota import mask_device_id
        from services.credit_service import _mask_user_id

        assert mask_device_id("a1b2c3d4e5f60718") == mask_user_id("a1b2c3d4e5f60718")
        assert _mask_user_id(USER_ID) == mask_user_id(USER_ID)
