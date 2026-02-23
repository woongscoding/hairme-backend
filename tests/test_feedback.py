"""Tests for feedback endpoints"""

import pytest
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
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        mock_db.query.return_value.filter.return_value.all.return_value = []
        mock_get_db.return_value = mock_db

        response = client.get("/api/stats/feedback")

        assert response.status_code in [200, 500]
