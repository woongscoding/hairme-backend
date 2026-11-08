"""Tests for Face Detection Service"""

import sys

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
    img = Image.new("RGB", (100, 100), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
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
        hue_value=15.0,
        confidence=0.95,
        face_features=[1.4, 120.0, 130.0, 110.0, 0.92, 0.85],
        skin_features=[45.0, 15.0],
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
        self, mock_mediapipe_analyzer, sample_image_data, mock_mp_features
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
        self, mock_mediapipe_analyzer, sample_image_data
    ):
        """MediaPipe 가 얼굴을 못 찾으면 Gemini 호출 없이 '얼굴 없음'을 돌려준다"""
        mock_mediapipe_analyzer.analyze.return_value = None

        fake_genai = MagicMock()
        with patch.dict(sys.modules, {"google.generativeai": fake_genai}):
            service = FaceDetectionService(mediapipe_analyzer=mock_mediapipe_analyzer)
            result = service.detect_face(sample_image_data)

        assert result == {
            "has_face": False,
            "face_count": 0,
            "method": "mediapipe",
            "features": None,
        }
        fake_genai.GenerativeModel.assert_not_called()

    def test_detect_face_mediapipe_raises_exception(
        self, mock_mediapipe_analyzer, sample_image_data
    ):
        """MediaPipe 예외도 '얼굴 없음'으로 처리하고 Gemini 를 부르지 않는다"""
        mock_mediapipe_analyzer.analyze.side_effect = Exception("MediaPipe error")

        fake_genai = MagicMock()
        with patch.dict(sys.modules, {"google.generativeai": fake_genai}):
            service = FaceDetectionService(mediapipe_analyzer=mock_mediapipe_analyzer)
            result = service.detect_face(sample_image_data)

        assert result["has_face"] is False
        assert result["face_count"] == 0
        assert result["features"] is None
        fake_genai.GenerativeModel.assert_not_called()

    def test_detect_face_no_analyzer_raises(self, sample_image_data):
        """분석기 미초기화는 서버 오류다 ('얼굴 없음'으로 감추지 않음)"""
        service = FaceDetectionService(mediapipe_analyzer=None)
        with pytest.raises(RuntimeError):
            service.detect_face(sample_image_data)

    def test_gemini_fallback_removed(self):
        """유료 Gemini 얼굴 검증 폴백은 더 이상 없다"""
        assert not hasattr(FaceDetectionService, "verify_face_with_gemini")
