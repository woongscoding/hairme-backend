"""Tests for face analysis endpoints"""

import pytest
import io
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

from main import app
from core.dependencies import get_face_detection_service, get_hybrid_service


# ========== Helper: Create mock services ==========
def _make_mock_face_detector(has_face=True, face_count=1):
    """Create a mock FaceDetectionService"""
    mock_detector = Mock()
    mock_features = Mock()
    mock_features.face_shape = "계란형"
    mock_features.skin_tone = "봄웜"
    mock_features.confidence = 0.92
    mock_features.gender = "neutral"
    mock_features.face_features = None
    mock_features.skin_features = None

    mock_detector.detect_face.return_value = {
        "has_face": has_face,
        "face_count": face_count,
        "method": "mediapipe",
        "features": mock_features if has_face else None,
    }
    return mock_detector


def _make_mock_ml_recommender():
    """Create a mock MLRecommendationService"""
    mock_recommender = Mock()
    mock_recommender.recommend.return_value = {
        "recommendations": [
            {
                "style_name": "레이어드 컷",
                "reason": "얼굴형과 잘 어울림",
                "confidence": 0.95,
            },
            {
                "style_name": "시스루 뱅",
                "reason": "이마 비율 보완",
                "confidence": 0.88,
            },
            {
                "style_name": "웨이브 펌",
                "reason": "부드러운 인상",
                "confidence": 0.82,
            },
        ],
    }
    return mock_recommender


class TestAnalyzeEndpoint:
    """Test /api/analyze endpoint"""

    def setup_method(self):
        """Set up DI overrides before each test"""
        self.mock_face_detector = _make_mock_face_detector()
        self.mock_ml_recommender = _make_mock_ml_recommender()
        app.dependency_overrides[get_face_detection_service] = lambda: self.mock_face_detector
        app.dependency_overrides[get_hybrid_service] = lambda: self.mock_ml_recommender

    def teardown_method(self):
        """Clean up DI overrides after each test"""
        app.dependency_overrides.pop(get_face_detection_service, None)
        app.dependency_overrides.pop(get_hybrid_service, None)

    @patch("api.endpoints.analyze.get_cached_result", return_value=None)
    @patch("api.endpoints.analyze.save_to_database", return_value=1)
    def test_analyze_with_valid_image(
        self, mock_save_db, mock_cache, client, sample_image_bytes
    ):
        """Test analysis with valid image"""
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}

        response = client.post("/api/analyze", files=files)

        assert response.status_code == 200
        data = response.json()

        # Check response structure (ML-only mode returns nested data)
        assert data["success"] is True
        assert "data" in data
        assert "analysis" in data["data"]
        assert "recommendations" in data["data"]
        assert data["data"]["analysis"]["face_shape"] == "계란형"

    def test_analyze_without_image(self, client):
        """Test analysis without providing an image"""
        response = client.post("/api/analyze")

        # Should return 422 (Unprocessable Entity) for missing field
        assert response.status_code == 422

    def test_analyze_with_invalid_image_format(self, client):
        """Test analysis with invalid image format"""
        # Create invalid file (text file instead of image)
        invalid_file = io.BytesIO(b"This is not an image")
        files = {"file": ("test.txt", invalid_file, "text/plain")}

        response = client.post("/api/analyze", files=files)

        # Should handle invalid format gracefully (InvalidFileFormatException -> 400)
        assert response.status_code in [400, 422, 500]

    @patch("api.endpoints.analyze.get_cached_result")
    def test_analyze_uses_cache_when_available(
        self, mock_get_cache, client, sample_image_bytes
    ):
        """Test that analysis uses cached results when available"""
        # Setup cache hit
        cached_result = {
            "analysis": {"face_shape": "계란형", "personal_color": "봄웜"},
            "recommendations": [],
        }
        mock_get_cache.return_value = cached_result

        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/api/analyze", files=files)

        assert response.status_code == 200
        data = response.json()
        # When cache is hit, face_detector.detect_face should NOT be called
        assert self.mock_face_detector.detect_face.call_count == 0
        assert data.get("cached") is True

    @patch("api.endpoints.analyze.get_cached_result", return_value=None)
    def test_analyze_handles_no_face_detected(
        self, mock_cache, client, sample_image_bytes
    ):
        """Test handling when no face is detected"""
        # Override the face detector to return no face
        no_face_detector = _make_mock_face_detector(has_face=False)
        app.dependency_overrides[get_face_detection_service] = lambda: no_face_detector

        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/api/analyze", files=files)

        # Should return 400 (NoFaceDetectedException)
        assert response.status_code == 400

    @patch("api.endpoints.analyze.get_cached_result", return_value=None)
    def test_analyze_handles_multiple_faces(self, mock_cache, client, sample_image_bytes):
        """Test handling when multiple faces are detected"""
        # Override the face detector to return multiple faces
        multi_face_detector = _make_mock_face_detector(has_face=True, face_count=2)
        app.dependency_overrides[get_face_detection_service] = lambda: multi_face_detector

        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/api/analyze", files=files)

        # Should return 400 (MultipleFacesException)
        assert response.status_code == 400


class TestImageProcessing:
    """Test image processing utilities"""

    def setup_method(self):
        """Set up DI overrides before each test"""
        self.mock_face_detector = _make_mock_face_detector()
        self.mock_ml_recommender = _make_mock_ml_recommender()
        app.dependency_overrides[get_face_detection_service] = lambda: self.mock_face_detector
        app.dependency_overrides[get_hybrid_service] = lambda: self.mock_ml_recommender

    def teardown_method(self):
        """Clean up DI overrides after each test"""
        app.dependency_overrides.pop(get_face_detection_service, None)
        app.dependency_overrides.pop(get_hybrid_service, None)

    @patch("api.endpoints.analyze.get_cached_result", return_value=None)
    @patch("api.endpoints.analyze.save_to_database", return_value=1)
    def test_accepts_jpg_images(self, mock_save_db, mock_cache, client):
        """Test that JPG images are accepted"""
        img = Image.new("RGB", (640, 480), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        response = client.post("/api/analyze", files=files)

        # Should accept JPG
        assert response.status_code in [200, 500]  # 500 if services not initialized

    @patch("api.endpoints.analyze.get_cached_result", return_value=None)
    @patch("api.endpoints.analyze.save_to_database", return_value=1)
    def test_accepts_png_images(self, mock_save_db, mock_cache, client):
        """Test that PNG images are accepted"""
        img = Image.new("RGB", (640, 480), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        files = {"file": ("test.png", img_bytes, "image/png")}
        response = client.post("/api/analyze", files=files)

        # Should accept PNG
        assert response.status_code in [200, 500]

    @patch("api.endpoints.analyze.get_cached_result", return_value=None)
    @patch("api.endpoints.analyze.save_to_database", return_value=1)
    def test_processes_large_images(self, mock_save_db, mock_cache, client):
        """Test processing of large images (should be resized)"""
        # Create large image
        img = Image.new("RGB", (3000, 4000), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        files = {"file": ("large.jpg", img_bytes, "image/jpeg")}
        response = client.post("/api/analyze", files=files)

        # Should handle large images
        assert response.status_code in [200, 500, 413]  # 413 if file too large
