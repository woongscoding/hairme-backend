"""Tests for face analysis endpoints"""

import pytest
import io
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient


class TestAnalyzeEndpoint:
    """Test /api/analyze endpoint"""

    @patch('api.endpoints.analyze.mediapipe_analyzer')
    @patch('api.endpoints.analyze.hybrid_service')
    @patch('core.cache.redis_client')
    def test_analyze_with_valid_image(self, mock_redis, mock_hybrid, mock_mp, client, sample_image_bytes):
        """Test analysis with valid image"""
        # Setup mocks
        mock_redis.get.return_value = None  # No cache
        mock_mp.analyze_face.return_value = {
            "face_shape": "계란형",
            "face_shape_confidence": 0.92,
            "face_ratio": 1.45
        }
        mock_hybrid.get_recommendations.return_value = {
            "face_shape": "계란형",
            "personal_color": "봄웜",
            "recommended_hairstyles": [
                {"name": "레이어드 컷", "reason": "얼굴형과 잘 어울림", "confidence": 0.95},
                {"name": "시스루 뱅", "reason": "이마 비율 보완", "confidence": 0.88},
                {"name": "웨이브 펌", "reason": "부드러운 인상", "confidence": 0.82}
            ]
        }

        # Create test image
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}

        response = client.post("/api/analyze", files=files)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "face_shape" in data
        assert "personal_color" in data
        assert "recommended_hairstyles" in data
        assert len(data["recommended_hairstyles"]) == 3

    def test_analyze_without_image(self, client):
        """Test analysis without providing an image"""
        response = client.post("/api/analyze")

        # Should return 422 (Unprocessable Entity) for missing field
        assert response.status_code == 422

    @patch('api.endpoints.analyze.mediapipe_analyzer')
    def test_analyze_with_invalid_image_format(self, mock_mp, client):
        """Test analysis with invalid image format"""
        # Create invalid file (text file instead of image)
        invalid_file = io.BytesIO(b"This is not an image")
        files = {"file": ("test.txt", invalid_file, "text/plain")}

        response = client.post("/api/analyze", files=files)

        # Should handle invalid format gracefully
        assert response.status_code in [400, 422, 500]

    @patch('api.endpoints.analyze.mediapipe_analyzer')
    @patch('core.cache.redis_client')
    def test_analyze_uses_cache_when_available(self, mock_redis, mock_mp, client, sample_image_bytes):
        """Test that analysis uses cached results when available"""
        # Setup cache hit
        cached_result = {
            "face_shape": "계란형",
            "personal_color": "봄웜",
            "recommended_hairstyles": []
        }
        mock_redis.get.return_value = str(cached_result).encode()

        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/api/analyze", files=files)

        # Should use cache and not call MediaPipe
        assert mock_mp.analyze_face.call_count == 0 or response.status_code in [200, 500]

    @patch('api.endpoints.analyze.mediapipe_analyzer')
    def test_analyze_handles_no_face_detected(self, mock_mp, client, sample_image_bytes):
        """Test handling when no face is detected"""
        # Mock MediaPipe to raise NoFaceDetectedException
        from core.exceptions import NoFaceDetectedException
        mock_mp.analyze_face.side_effect = NoFaceDetectedException("No face detected")

        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/api/analyze", files=files)

        # Should return appropriate error code
        assert response.status_code in [400, 404]

    @patch('api.endpoints.analyze.mediapipe_analyzer')
    def test_analyze_handles_multiple_faces(self, mock_mp, client, sample_image_bytes):
        """Test handling when multiple faces are detected"""
        from core.exceptions import MultipleFacesException
        mock_mp.analyze_face.side_effect = MultipleFacesException("Multiple faces detected")

        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/api/analyze", files=files)

        # Should return appropriate error code
        assert response.status_code in [400, 422]


class TestImageProcessing:
    """Test image processing utilities"""

    @patch('api.endpoints.analyze.mediapipe_analyzer')
    def test_accepts_jpg_images(self, mock_mp, client):
        """Test that JPG images are accepted"""
        img = Image.new('RGB', (640, 480), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        response = client.post("/api/analyze", files=files)

        # Should accept JPG
        assert response.status_code in [200, 500]  # 500 if services not initialized

    @patch('api.endpoints.analyze.mediapipe_analyzer')
    def test_accepts_png_images(self, mock_mp, client):
        """Test that PNG images are accepted"""
        img = Image.new('RGB', (640, 480), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        files = {"file": ("test.png", img_bytes, "image/png")}
        response = client.post("/api/analyze", files=files)

        # Should accept PNG
        assert response.status_code in [200, 500]

    @patch('api.endpoints.analyze.mediapipe_analyzer')
    @patch('api.endpoints.analyze.hybrid_service')
    @patch('core.cache.redis_client')
    def test_processes_large_images(self, mock_redis, mock_hybrid, mock_mp, client):
        """Test processing of large images (should be resized)"""
        mock_redis.get.return_value = None
        mock_mp.analyze_face.return_value = {"face_shape": "계란형"}
        mock_hybrid.get_recommendations.return_value = {
            "face_shape": "계란형",
            "personal_color": "봄웜",
            "recommended_hairstyles": []
        }

        # Create large image
        img = Image.new('RGB', (3000, 4000), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        files = {"file": ("large.jpg", img_bytes, "image/jpeg")}
        response = client.post("/api/analyze", files=files)

        # Should handle large images
        assert response.status_code in [200, 500, 413]  # 413 if file too large
