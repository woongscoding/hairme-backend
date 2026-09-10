"""Tests for feedback endpoints (DynamoDB-only backend)"""

import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def _clean_public_stats_cache():
    """공개 통계 TTL 캐시는 프로세스 전역이므로 매 테스트마다 비운다"""
    import api.endpoints.feedback as feedback_module

    feedback_module._public_stats_cache.clear()
    yield
    feedback_module._public_stats_cache.clear()


def _stats_payload(total=10, feedback_count=5, recent=None):
    return {
        "success": True,
        "total_analysis": total,
        "total_feedback": feedback_count,
        "like_counts": {"style_1": 1, "style_2": 0, "style_3": 0},
        "dislike_counts": {"style_1": 0, "style_2": 1, "style_3": 0},
        "recent_feedbacks": recent if recent is not None else [],
    }


class TestFeedbackEndpoint:
    """Test /api/feedback endpoint"""

    @patch("api.endpoints.feedback.save_feedback")
    @patch("api.endpoints.feedback.get_analysis")
    def test_submit_positive_feedback(self, mock_get_analysis, mock_save, client):
        """Test submitting positive feedback"""
        mock_get_analysis.return_value = {"analysis_id": "abc-123"}
        mock_save.return_value = True

        feedback_data = {
            "analysis_id": "abc-123",
            "style_index": 1,
            "feedback": "good",
            "naver_clicked": False,
        }

        response = client.post("/api/feedback", json=feedback_data)

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["style_index"] == 1
        # 저장 시 정규화된 값이 전달되어야 함
        assert mock_save.call_args.kwargs["feedback"] == "good"
        assert mock_save.call_args.kwargs["analysis_id"] == "abc-123"

    @patch("api.endpoints.feedback.save_feedback")
    @patch("api.endpoints.feedback.get_analysis")
    def test_submit_negative_feedback(self, mock_get_analysis, mock_save, client):
        """Test submitting negative feedback"""
        mock_get_analysis.return_value = {"analysis_id": "abc-123"}
        mock_save.return_value = True

        feedback_data = {
            "analysis_id": "abc-123",
            "style_index": 2,
            "feedback": "bad",
            "naver_clicked": False,
        }

        response = client.post("/api/feedback", json=feedback_data)

        assert response.status_code == 200
        assert response.json()["style_index"] == 2
        assert mock_save.call_args.kwargs["feedback"] == "bad"

    def test_submit_feedback_without_required_fields(self, client):
        """Test submitting feedback without required fields"""
        incomplete_data = {"naver_clicked": True}

        response = client.post("/api/feedback", json=incomplete_data)

        # Should return 422 for missing fields
        assert response.status_code == 422

    @patch("api.endpoints.feedback.save_feedback")
    @patch("api.endpoints.feedback.get_analysis")
    def test_submit_feedback_with_invalid_style_index(
        self, mock_get_analysis, mock_save, client
    ):
        """Test submitting feedback with invalid style_index (outside 1-5 range)"""
        invalid_data = {
            "analysis_id": "abc-123",
            "style_index": 10,  # Invalid: should be 1-5
            "feedback": "good",
            "naver_clicked": False,
        }

        response = client.post("/api/feedback", json=invalid_data)

        # Should reject invalid style_index (Pydantic validation -> 422)
        assert response.status_code in (400, 422)
        mock_save.assert_not_called()

    @patch("api.endpoints.feedback.save_feedback")
    @patch("api.endpoints.feedback.get_analysis")
    def test_submit_feedback_analysis_not_found(
        self, mock_get_analysis, mock_save, client
    ):
        """Test submitting feedback when analysis record is not found"""
        mock_get_analysis.return_value = None

        feedback_data = {
            "analysis_id": "missing-id",
            "style_index": 1,
            "feedback": "good",
            "naver_clicked": False,
        }

        response = client.post("/api/feedback", json=feedback_data)

        # Should return 404 for missing analysis record
        assert response.status_code == 404
        mock_save.assert_not_called()

    def test_submit_feedback_with_null_analysis_id(self, client):
        """Test submitting feedback with null analysis_id"""
        feedback_data = {
            "analysis_id": None,
            "style_index": 1,
            "feedback": "good",
            "naver_clicked": False,
        }

        response = client.post("/api/feedback", json=feedback_data)

        # Should return 400 for null analysis_id
        assert response.status_code in (400, 422)


class TestFeedbackStats:
    """Test feedback statistics endpoint"""

    @patch("api.endpoints.feedback.get_dynamodb_stats")
    def test_get_feedback_stats(self, mock_stats, client):
        """Test retrieving feedback statistics"""
        mock_stats.return_value = _stats_payload()

        response = client.get("/api/stats/feedback")

        assert response.status_code == 200
        data = response.json()
        assert data["total_analysis"] == 10
        assert data["total_feedback"] == 5

    @patch("api.endpoints.feedback.get_dynamodb_stats")
    def test_public_stats_never_expose_identifiers(self, mock_stats, client):
        """Public stats must not leak analysis ids (would allow feedback spoofing)"""
        mock_stats.return_value = _stats_payload(
            recent=[
                {
                    "id": "abc-123",
                    "analysis_id": "abc-123",
                    "image_hash": "deadbeef",
                    "user_id": "u1",
                    "face_shape": "계란형",
                    "style_1_feedback": "good",
                }
            ]
        )

        response = client.get("/api/stats/feedback")

        assert response.status_code == 200
        data = response.json()

        assert "total_analysis" in data
        assert "like_counts" in data
        assert data["recent_feedbacks"]
        for item in data["recent_feedbacks"]:
            assert "id" not in item
            assert "analysis_id" not in item
            assert "image_hash" not in item
            assert "user_id" not in item
            assert item["face_shape"] == "계란형"

    @patch("api.endpoints.feedback.get_dynamodb_stats")
    def test_stats_failure_returns_500(self, mock_stats, client):
        """DynamoDB 통계 조회 실패 시 500 (내부 메시지 노출 없음)"""
        mock_stats.return_value = {"success": False}

        response = client.get("/api/stats/feedback")

        assert response.status_code == 500


class TestPublicStatsSanitizer:
    """Unit tests for the public stats sanitizer / TTL cache"""

    def test_strip_public_identifiers_removes_ids_keeps_aggregates(self):
        from api.endpoints.feedback import _strip_public_identifiers

        raw = {
            "success": True,
            "total_analysis": 3,
            "total_feedback": 2,
            "like_counts": {"style_1": 1},
            "dislike_counts": {"style_1": 0},
            "recent_feedbacks": [
                {
                    "id": "abc-123",
                    "analysis_id": "abc-123",
                    "image_hash": "deadbeef",
                    "face_shape": "계란형",
                    "style_1_feedback": "good",
                }
            ],
        }

        cleaned = _strip_public_identifiers(raw)

        assert cleaned["total_analysis"] == 3
        assert cleaned["like_counts"] == {"style_1": 1}
        assert cleaned["recent_feedbacks"] == [
            {"face_shape": "계란형", "style_1_feedback": "good"}
        ]
        # 원본은 변경되지 않아야 함 (admin 엔드포인트는 원본 사용)
        assert raw["recent_feedbacks"][0]["id"] == "abc-123"

    def test_ttl_cache_roundtrip_and_expiry(self):
        import api.endpoints.feedback as feedback_module

        key = feedback_module._STATS_CACHE_KEY
        assert key == "dynamodb"

        assert feedback_module._get_cached_public_stats(key) is None

        feedback_module._set_cached_public_stats(key, {"success": True})
        assert feedback_module._get_cached_public_stats(key) == {"success": True}

        # 만료 시뮬레이션
        feedback_module._public_stats_cache[key]["ts"] -= (
            feedback_module._PUBLIC_STATS_CACHE_TTL_SECONDS + 1
        )
        assert feedback_module._get_cached_public_stats(key) is None

    @patch("api.endpoints.feedback.get_dynamodb_stats")
    def test_stats_endpoint_uses_cache(self, mock_stats, client):
        """Second call within the TTL must not re-run the table scan"""
        mock_stats.return_value = _stats_payload(total=7, feedback_count=3)

        first = client.get("/api/stats/feedback")
        second = client.get("/api/stats/feedback")

        assert first.status_code == 200
        assert second.json() == first.json()
        # DB 스캔은 첫 요청에서만 수행되어야 함
        assert mock_stats.call_count == 1
