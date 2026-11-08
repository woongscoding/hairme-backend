"""Tests for feedback endpoints (DynamoDB-only backend)"""

from unittest.mock import patch


class TestFeedbackEndpoint:
    """Test /api/feedback/submit endpoint"""

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

        response = client.post("/api/feedback/submit", json=feedback_data)

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

        response = client.post("/api/feedback/submit", json=feedback_data)

        assert response.status_code == 200
        assert response.json()["style_index"] == 2
        assert mock_save.call_args.kwargs["feedback"] == "bad"

    def test_submit_feedback_without_required_fields(self, client):
        """Test submitting feedback without required fields"""
        incomplete_data = {"naver_clicked": True}

        response = client.post("/api/feedback/submit", json=incomplete_data)

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

        response = client.post("/api/feedback/submit", json=invalid_data)

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

        response = client.post("/api/feedback/submit", json=feedback_data)

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

        response = client.post("/api/feedback/submit", json=feedback_data)

        # Should return 400 for null analysis_id
        assert response.status_code in (400, 422)
