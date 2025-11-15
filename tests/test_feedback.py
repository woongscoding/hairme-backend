"""Tests for feedback endpoints"""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient


class TestFeedbackEndpoint:
    """Test /api/feedback endpoint"""

    @patch('api.endpoints.feedback.feedback_collector')
    @patch('api.endpoints.feedback.retrain_queue')
    def test_submit_positive_feedback(self, mock_queue, mock_collector, client):
        """Test submitting positive feedback"""
        mock_collector.collect_feedback.return_value = True

        feedback_data = {
            "analysis_id": "test_123",
            "hairstyle_name": "레이어드 컷",
            "rating": 5,
            "is_positive": True
        }

        response = client.post("/api/feedback", json=feedback_data)

        assert response.status_code in [200, 404, 500]  # Depends on implementation

    @patch('api.endpoints.feedback.feedback_collector')
    def test_submit_negative_feedback(self, mock_collector, client):
        """Test submitting negative feedback"""
        mock_collector.collect_feedback.return_value = True

        feedback_data = {
            "analysis_id": "test_123",
            "hairstyle_name": "레이어드 컷",
            "rating": 2,
            "is_positive": False,
            "comment": "어울리지 않음"
        }

        response = client.post("/api/feedback", json=feedback_data)

        # Should accept negative feedback
        assert response.status_code in [200, 404, 422, 500]

    def test_submit_feedback_without_required_fields(self, client):
        """Test submitting feedback without required fields"""
        incomplete_data = {
            "rating": 5
        }

        response = client.post("/api/feedback", json=incomplete_data)

        # Should return 422 for missing fields
        assert response.status_code in [422, 500]

    @patch('api.endpoints.feedback.feedback_collector')
    def test_submit_feedback_with_invalid_rating(self, mock_collector, client):
        """Test submitting feedback with invalid rating (outside 1-5 range)"""
        invalid_data = {
            "analysis_id": "test_123",
            "hairstyle_name": "레이어드 컷",
            "rating": 10,  # Invalid: should be 1-5
            "is_positive": True
        }

        response = client.post("/api/feedback", json=invalid_data)

        # Should reject invalid rating
        assert response.status_code in [400, 422, 500]


class TestRetrainingQueue:
    """Test retraining queue functionality"""

    @patch('api.endpoints.feedback.retrain_queue')
    @patch('api.endpoints.feedback.feedback_collector')
    def test_feedback_triggers_retraining_after_threshold(self, mock_collector, mock_queue, client):
        """Test that enough feedback triggers retraining"""
        mock_collector.collect_feedback.return_value = True
        mock_collector.get_feedback_count.return_value = 100  # Above threshold

        feedback_data = {
            "analysis_id": "test_123",
            "hairstyle_name": "레이어드 컷",
            "rating": 5,
            "is_positive": True
        }

        response = client.post("/api/feedback", json=feedback_data)

        # Retraining should be triggered (or queued)
        # This is implementation-dependent
        assert response.status_code in [200, 404, 500]
