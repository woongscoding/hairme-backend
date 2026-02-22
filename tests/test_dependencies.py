"""Tests for dependency injection"""

import pytest
from unittest.mock import Mock

from core.dependencies import (
    init_services,
    get_face_detection_service,
    get_hybrid_service,
)
from core import dependencies
from models.mediapipe_analyzer import MediaPipeFaceAnalyzer
from services.face_detection_service import FaceDetectionService


@pytest.fixture(autouse=True)
def reset_dependencies():
    """Reset global dependencies before each test"""
    dependencies._mediapipe_analyzer = None
    dependencies._face_detection_service = None
    dependencies._hybrid_service = None

    # Clear lru_cache
    get_face_detection_service.cache_clear()
    get_hybrid_service.cache_clear()

    yield


class TestDependencies:
    """Test suite for dependency injection functions"""

    def test_init_services_initializes_all_services(self):
        """Test that init_services initializes services"""
        mock_analyzer = Mock(spec=MediaPipeFaceAnalyzer)
        mock_hybrid = Mock()

        init_services(
            mediapipe_analyzer=mock_analyzer,
            hybrid_service=mock_hybrid,
        )

        assert dependencies._mediapipe_analyzer is mock_analyzer
        assert dependencies._hybrid_service is mock_hybrid
        assert isinstance(dependencies._face_detection_service, FaceDetectionService)

    def test_get_face_detection_service_after_init(self):
        """Test that get_face_detection_service returns instance after init"""
        mock_analyzer = Mock(spec=MediaPipeFaceAnalyzer)
        init_services(mediapipe_analyzer=mock_analyzer)

        service = get_face_detection_service()
        assert isinstance(service, FaceDetectionService)

    def test_get_hybrid_service_after_init(self):
        """Test that get_hybrid_service returns instance after init"""
        mock_hybrid = Mock()
        init_services(hybrid_service=mock_hybrid)

        service = get_hybrid_service()
        assert service is mock_hybrid

    def test_services_cached_with_lru_cache(self):
        """Test that services are cached using lru_cache"""
        mock_hybrid = Mock()
        init_services(hybrid_service=mock_hybrid)

        service1 = get_hybrid_service()
        service2 = get_hybrid_service()

        assert service1 is service2
