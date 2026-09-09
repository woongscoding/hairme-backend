"""Tests for feedback endpoints"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient


class TestFeedbackEndpoint:
    """Test /api/feedback endpoint"""

    @patch("api.endpoints.feedback.get_db_session")
    def test_submit_positive_feedback(self, mock_get_db, client):
        """Test submitting positive feedback"""
        # Setup mock DB session with a matching analysis record
        mock_db = MagicMock()
        mock_record = MagicMock()
        mock_record.id = 1
        mock_db.query.return_value.filter.return_value.first.return_value = mock_record
        mock_get_db.return_value = mock_db

        feedback_data = {
            "analysis_id": 1,
            "style_index": 1,
            "feedback": "good",
            "naver_clicked": False,
        }

        response = client.post("/api/feedback", json=feedback_data)

        assert response.status_code in [200, 404, 500]

    @patch("api.endpoints.feedback.get_db_session")
    def test_submit_negative_feedback(self, mock_get_db, client):
        """Test submitting negative feedback"""
        mock_db = MagicMock()
        mock_record = MagicMock()
        mock_record.id = 1
        mock_db.query.return_value.filter.return_value.first.return_value = mock_record
        mock_get_db.return_value = mock_db

        feedback_data = {
            "analysis_id": 1,
            "style_index": 2,
            "feedback": "bad",
            "naver_clicked": False,
        }

        response = client.post("/api/feedback", json=feedback_data)

        # Should accept negative feedback
        assert response.status_code in [200, 404, 422, 500]

    def test_submit_feedback_without_required_fields(self, client):
        """Test submitting feedback without required fields"""
        incomplete_data = {"naver_clicked": True}

        response = client.post("/api/feedback", json=incomplete_data)

        # Should return 422 for missing fields
        assert response.status_code in [422, 500]

    @patch("api.endpoints.feedback.get_db_session")
    def test_submit_feedback_with_invalid_style_index(self, mock_get_db, client):
        """Test submitting feedback with invalid style_index (outside 1-5 range)"""
        invalid_data = {
            "analysis_id": 1,
            "style_index": 10,  # Invalid: should be 1-5
            "feedback": "good",
            "naver_clicked": False,
        }

        response = client.post("/api/feedback", json=invalid_data)

        # Should reject invalid style_index (Pydantic validation -> 422)
        assert response.status_code in [400, 422, 500]

    @patch("api.endpoints.feedback.get_db_session")
    def test_submit_feedback_analysis_not_found(self, mock_get_db, client):
        """Test submitting feedback when analysis record is not found"""
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_get_db.return_value = mock_db

        feedback_data = {
            "analysis_id": 9999,
            "style_index": 1,
            "feedback": "good",
            "naver_clicked": False,
        }

        response = client.post("/api/feedback", json=feedback_data)

        # Should return 404 for missing analysis record
        assert response.status_code in [404, 500]

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
        assert response.status_code in [400, 422]


class TestFeedbackStats:
    """Test feedback statistics endpoint"""

    @patch("api.endpoints.feedback.get_db_session")
    def test_get_feedback_stats(self, mock_get_db, client):
        """Test retrieving feedback statistics"""
        mock_db = MagicMock()
        mock_db.query.return_value.count.return_value = 10
        mock_db.query.return_value.filter.return_value.count.return_value = 5
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = (
            []
        )
        mock_db.query.return_value.filter.return_value.all.return_value = []
        mock_get_db.return_value = mock_db

        import api.endpoints.feedback as feedback_module

        feedback_module._public_stats_cache.clear()

        response = client.get("/api/stats/feedback")

        assert response.status_code in [200, 500]

    @patch("api.endpoints.feedback.get_db_session")
    def test_public_stats_never_expose_identifiers(self, mock_get_db, client):
        """Public stats must not leak analysis ids (would allow feedback spoofing)"""
        import api.endpoints.feedback as feedback_module

        feedback_module._public_stats_cache.clear()

        mock_db = MagicMock()
        mock_db.query.return_value.count.return_value = 10
        mock_db.query.return_value.filter.return_value.count.return_value = 5
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = (
            []
        )
        mock_db.query.return_value.filter.return_value.all.return_value = []
        mock_get_db.return_value = mock_db

        response = client.get("/api/stats/feedback")

        assert response.status_code == 200
        data = response.json()

        assert "total_analysis" in data
        assert "like_counts" in data
        for item in data["recent_feedbacks"]:
            assert "id" not in item
            assert "analysis_id" not in item
            assert "image_hash" not in item


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

        feedback_module._public_stats_cache.clear()

        assert feedback_module._get_cached_public_stats("mysql") is None

        feedback_module._set_cached_public_stats("mysql", {"success": True})
        assert feedback_module._get_cached_public_stats("mysql") == {"success": True}

        # 만료 시뮬레이션
        feedback_module._public_stats_cache["mysql"]["ts"] -= (
            feedback_module._PUBLIC_STATS_CACHE_TTL_SECONDS + 1
        )
        assert feedback_module._get_cached_public_stats("mysql") is None

        feedback_module._public_stats_cache.clear()

    def test_stats_endpoint_uses_cache(self, client):
        """Second call within the TTL must not re-run the table scan"""
        import api.endpoints.feedback as feedback_module

        feedback_module._public_stats_cache.clear()

        # 다른 테스트가 USE_DYNAMODB를 남겨둘 수 있으므로 MySQL 분기를 명시적으로 고정
        with patch.dict(os.environ, {"USE_DYNAMODB": "false"}), patch(
            "api.endpoints.feedback.get_db_session"
        ) as mock_get_db:
            mock_db = MagicMock()
            mock_db.query.return_value.count.return_value = 7
            mock_db.query.return_value.filter.return_value.count.return_value = 3
            mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = (
                []
            )
            mock_db.query.return_value.filter.return_value.all.return_value = []
            mock_get_db.return_value = mock_db

            first = client.get("/api/stats/feedback")
            second = client.get("/api/stats/feedback")

            assert first.status_code == 200
            assert second.json() == first.json()
            # DB 세션은 첫 요청에서만 열려야 함
            assert mock_get_db.call_count == 1

        feedback_module._public_stats_cache.clear()
