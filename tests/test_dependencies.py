"""Tests for dependency injection"""

import pytest
from unittest.mock import Mock, patch

from core.dependencies import (
    get_face_detection_service,
    get_hybrid_service,
    get_mediapipe_analyzer,
)
from core import dependencies
from models.mediapipe_analyzer import MediaPipeFaceAnalyzer
from services.face_detection_service import FaceDetectionService


@pytest.fixture(autouse=True)
def reset_dependencies(monkeypatch):
    """Reset global singletons for each test (restored afterwards)"""
    monkeypatch.setattr(dependencies, "_mediapipe_analyzer", None)
    monkeypatch.setattr(dependencies, "_face_detection_service", None)
    monkeypatch.setattr(dependencies, "_hybrid_service", None)
    yield


class TestDependencies:
    """Test suite for dependency injection functions"""

    def test_get_face_detection_service_uses_existing_analyzer(self):
        """Face detection service is built on the (already loaded) analyzer"""
        mock_analyzer = Mock(spec=MediaPipeFaceAnalyzer)
        dependencies._mediapipe_analyzer = mock_analyzer

        service = get_face_detection_service()

        assert isinstance(service, FaceDetectionService)
        assert service.mediapipe_analyzer is mock_analyzer
        assert dependencies._face_detection_service is service

    def test_get_mediapipe_analyzer_lazy_loads_once(self):
        """Analyzer is created on first call only and stored in the module global"""
        with patch("models.mediapipe_analyzer.MediaPipeFaceAnalyzer") as analyzer_cls:
            first = get_mediapipe_analyzer()
            second = get_mediapipe_analyzer()

        analyzer_cls.assert_called_once_with()
        assert first is second
        assert dependencies._mediapipe_analyzer is first

    def test_get_hybrid_service_lazy_loads_once(self):
        """Recommendation service is created once and then reused"""
        mock_hybrid = Mock()
        with patch(
            "services.hybrid_recommender.get_ml_recommendation_service",
            return_value=mock_hybrid,
        ) as factory:
            service1 = get_hybrid_service()
            service2 = get_hybrid_service()

        factory.assert_called_once_with()
        assert service1 is mock_hybrid
        assert service2 is mock_hybrid

    def test_get_hybrid_service_failure_is_not_cached(self):
        """A failed load raises RuntimeError and is retried on the next call"""
        mock_hybrid = Mock()
        with patch(
            "services.hybrid_recommender.get_ml_recommendation_service",
            side_effect=[Exception("boom"), mock_hybrid],
        ):
            with pytest.raises(RuntimeError):
                get_hybrid_service()
            assert get_hybrid_service() is mock_hybrid
