"""Tests for Face Detection Service"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from PIL import Image
import io

from services.face_detection_service import FaceDetectionService
from models.mediapipe_analyzer import MediaPipeFaceFeatures


@pytest.fixture
def mock_mediapipe_analyzer():
    """Mock MediaPipe analyzer"""
    analyzer = Mock()
    return analyzer


@pytest.fixture
def sample_image_data():
    """Generate sample image bytes"""
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    return img_bytes.getvalue()


@pytest.fixture
def mock_mp_features():
    """Mock MediaPipe features"""
    return MediaPipeFaceFeatures(
        face_shape="계란형",
        skin_tone="밝은 톤",
        face_ratio=1.4,
        forehead_width=120.0,
        cheekbone_width=130.0,
        jaw_width=110.0,
        ITA_value=45.0,
        confidence=0.95
    )


class TestFaceDetectionService:
    """Test suite for FaceDetectionService"""

    def test_init_with_analyzer(self, mock_mediapipe_analyzer):
        """Test initialization with analyzer"""
        service = FaceDetectionService(mediapipe_analyzer=mock_mediapipe_analyzer)
        assert service.mediapipe_analyzer is mock_mediapipe_analyzer

    def test_init_without_analyzer(self):
        """Test initialization without analyzer"""
        service = FaceDetectionService()
        assert service.mediapipe_analyzer is None

    def test_detect_face_with_mediapipe_success(
        self,
        mock_mediapipe_analyzer,
        sample_image_data,
        mock_mp_features
    ):
        """Test successful face detection with MediaPipe"""
        mock_mediapipe_analyzer.analyze.return_value = mock_mp_features

        service = FaceDetectionService(mediapipe_analyzer=mock_mediapipe_analyzer)
        result = service.detect_face(sample_image_data)

        assert result["has_face"] is True
        assert result["face_count"] == 1
        assert result["method"] == "mediapipe"
        assert result["features"] == mock_mp_features
        mock_mediapipe_analyzer.analyze.assert_called_once_with(sample_image_data)

    def test_detect_face_mediapipe_returns_none(
        self,
        mock_mediapipe_analyzer,
        sample_image_data
    ):
        """Test fallback to Gemini when MediaPipe returns None"""
        mock_mediapipe_analyzer.analyze.return_value = None

        with patch.object(
            FaceDetectionService,
            'verify_face_with_gemini'
        ) as mock_gemini:
            mock_gemini.return_value = {
                "has_face": True,
                "face_count": 1,
                "method": "gemini"
            }

            service = FaceDetectionService(mediapipe_analyzer=mock_mediapipe_analyzer)
            result = service.detect_face(sample_image_data)

            assert result["method"] == "gemini"
            mock_gemini.assert_called_once_with(sample_image_data)

    def test_detect_face_mediapipe_raises_exception(
        self,
        mock_mediapipe_analyzer,
        sample_image_data
    ):
        """Test fallback to Gemini when MediaPipe raises exception"""
        mock_mediapipe_analyzer.analyze.side_effect = Exception("MediaPipe error")

        with patch.object(
            FaceDetectionService,
            'verify_face_with_gemini'
        ) as mock_gemini:
            mock_gemini.return_value = {
                "has_face": True,
                "face_count": 1,
                "method": "gemini"
            }

            service = FaceDetectionService(mediapipe_analyzer=mock_mediapipe_analyzer)
            result = service.detect_face(sample_image_data)

            assert result["method"] == "gemini"
            mock_gemini.assert_called_once()

    def test_detect_face_no_analyzer_uses_gemini(self, sample_image_data):
        """Test direct Gemini usage when no analyzer provided"""
        with patch.object(
            FaceDetectionService,
            'verify_face_with_gemini'
        ) as mock_gemini:
            mock_gemini.return_value = {
                "has_face": False,
                "face_count": 0,
                "method": "gemini"
            }

            service = FaceDetectionService(mediapipe_analyzer=None)
            result = service.detect_face(sample_image_data)

            assert result["method"] == "gemini"
            mock_gemini.assert_called_once_with(sample_image_data)

    @patch('services.face_detection_service.genai.GenerativeModel')
    def test_verify_face_with_gemini_success(
        self,
        mock_genai_model,
        sample_image_data
    ):
        """Test successful Gemini face verification"""
        # Mock Gemini response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"has_face": true, "face_count": 1}'
        mock_model.generate_content.return_value = mock_response
        mock_genai_model.return_value = mock_model

        service = FaceDetectionService()
        result = service.verify_face_with_gemini(sample_image_data)

        assert result["has_face"] is True
        assert result["face_count"] == 1
        assert result["method"] == "gemini"
        assert "error" not in result

    @patch('services.face_detection_service.genai.GenerativeModel')
    def test_verify_face_with_gemini_no_face(
        self,
        mock_genai_model,
        sample_image_data
    ):
        """Test Gemini verification with no face detected"""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"has_face": false, "face_count": 0}'
        mock_model.generate_content.return_value = mock_response
        mock_genai_model.return_value = mock_model

        service = FaceDetectionService()
        result = service.verify_face_with_gemini(sample_image_data)

        assert result["has_face"] is False
        assert result["face_count"] == 0
        assert result["method"] == "gemini"

    @patch('services.face_detection_service.genai.GenerativeModel')
    def test_verify_face_with_gemini_error(
        self,
        mock_genai_model,
        sample_image_data
    ):
        """Test Gemini verification error handling"""
        mock_genai_model.side_effect = Exception("API Error")

        service = FaceDetectionService()
        result = service.verify_face_with_gemini(sample_image_data)

        assert result["has_face"] is False
        assert result["face_count"] == 0
        assert result["method"] == "gemini"
        assert "error" in result
        assert "API Error" in result["error"]

    @patch('services.face_detection_service.genai.GenerativeModel')
    def test_verify_face_with_gemini_invalid_json(
        self,
        mock_genai_model,
        sample_image_data
    ):
        """Test Gemini verification with invalid JSON response"""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = 'Invalid JSON'
        mock_model.generate_content.return_value = mock_response
        mock_genai_model.return_value = mock_model

        service = FaceDetectionService()
        result = service.verify_face_with_gemini(sample_image_data)

        assert result["has_face"] is False
        assert result["face_count"] == 0
        assert "error" in result
