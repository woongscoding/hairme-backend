"""Tests for dependency injection"""

import pytest
from unittest.mock import Mock, MagicMock

from core.dependencies import (
    init_services,
    get_face_detection_service,
    get_gemini_analysis_service,
    get_hybrid_service,
    get_feedback_collector,
    get_retrain_queue
)
from core import dependencies
from models.mediapipe_analyzer import MediaPipeFaceAnalyzer
from services.face_detection_service import FaceDetectionService
from services.gemini_analysis_service import GeminiAnalysisService
from services.hybrid_recommender import HybridRecommendationService
from services.feedback_collector import FeedbackCollector
from services.retrain_queue import RetrainQueue


@pytest.fixture(autouse=True)
def reset_dependencies():
    """Reset global dependencies before each test"""
    # Reset all global variables
    dependencies._mediapipe_analyzer = None
    dependencies._face_detection_service = None
    dependencies._gemini_analysis_service = None
    dependencies._hybrid_service = None
    dependencies._feedback_collector = None
    dependencies._retrain_queue = None

    # Clear lru_cache
    get_face_detection_service.cache_clear()
    get_gemini_analysis_service.cache_clear()
    get_hybrid_service.cache_clear()
    get_feedback_collector.cache_clear()
    get_retrain_queue.cache_clear()

    yield


class TestDependencies:
    """Test suite for dependency injection functions"""

    def test_init_services_initializes_all_services(self):
        """Test that init_services initializes all services"""
        mock_analyzer = Mock(spec=MediaPipeFaceAnalyzer)
        mock_hybrid = Mock(spec=HybridRecommendationService)
        mock_feedback = Mock(spec=FeedbackCollector)
        mock_retrain = Mock(spec=RetrainQueue)

        init_services(
            mediapipe_analyzer=mock_analyzer,
            hybrid_service=mock_hybrid,
            feedback_collector=mock_feedback,
            retrain_queue=mock_retrain
        )

        assert dependencies._mediapipe_analyzer is mock_analyzer
        assert dependencies._hybrid_service is mock_hybrid
        assert dependencies._feedback_collector is mock_feedback
        assert dependencies._retrain_queue is mock_retrain
        assert isinstance(dependencies._face_detection_service, FaceDetectionService)
        assert isinstance(dependencies._gemini_analysis_service, GeminiAnalysisService)

    def test_get_face_detection_service_after_init(self):
        """Test that get_face_detection_service returns instance after init"""
        mock_analyzer = Mock(spec=MediaPipeFaceAnalyzer)
        init_services(mediapipe_analyzer=mock_analyzer)

        service = get_face_detection_service()
        assert isinstance(service, FaceDetectionService)
        assert service.mediapipe_analyzer is mock_analyzer

    def test_get_face_detection_service_raises_before_init(self):
        """Test that get_face_detection_service raises error before init"""
        with pytest.raises(RuntimeError) as exc_info:
            get_face_detection_service()

        assert "FaceDetectionService not initialized" in str(exc_info.value)

    def test_get_gemini_analysis_service_after_init(self):
        """Test that get_gemini_analysis_service returns instance after init"""
        init_services()

        service = get_gemini_analysis_service()
        assert isinstance(service, GeminiAnalysisService)
        assert service.max_retries == 3

    def test_get_gemini_analysis_service_raises_before_init(self):
        """Test that get_gemini_analysis_service raises error before init"""
        with pytest.raises(RuntimeError) as exc_info:
            get_gemini_analysis_service()

        assert "GeminiAnalysisService not initialized" in str(exc_info.value)

    def test_get_hybrid_service_after_init(self):
        """Test that get_hybrid_service returns instance after init"""
        mock_hybrid = Mock(spec=HybridRecommendationService)
        init_services(hybrid_service=mock_hybrid)

        service = get_hybrid_service()
        assert service is mock_hybrid

    def test_get_hybrid_service_raises_before_init(self):
        """Test that get_hybrid_service raises error before init"""
        with pytest.raises(RuntimeError) as exc_info:
            get_hybrid_service()

        assert "HybridRecommendationService not initialized" in str(exc_info.value)

    def test_get_feedback_collector_after_init(self):
        """Test that get_feedback_collector returns instance after init"""
        mock_feedback = Mock(spec=FeedbackCollector)
        init_services(feedback_collector=mock_feedback)

        service = get_feedback_collector()
        assert service is mock_feedback

    def test_get_feedback_collector_raises_before_init(self):
        """Test that get_feedback_collector raises error before init"""
        with pytest.raises(RuntimeError) as exc_info:
            get_feedback_collector()

        assert "FeedbackCollector not initialized" in str(exc_info.value)

    def test_get_retrain_queue_after_init(self):
        """Test that get_retrain_queue returns instance after init"""
        mock_retrain = Mock(spec=RetrainQueue)
        init_services(retrain_queue=mock_retrain)

        service = get_retrain_queue()
        assert service is mock_retrain

    def test_get_retrain_queue_raises_before_init(self):
        """Test that get_retrain_queue raises error before init"""
        with pytest.raises(RuntimeError) as exc_info:
            get_retrain_queue()

        assert "RetrainQueue not initialized" in str(exc_info.value)

    def test_services_cached_with_lru_cache(self):
        """Test that services are cached using lru_cache"""
        mock_hybrid = Mock(spec=HybridRecommendationService)
        init_services(hybrid_service=mock_hybrid)

        service1 = get_hybrid_service()
        service2 = get_hybrid_service()

        # Should return same instance (cached)
        assert service1 is service2
