"""Dependency injection providers for FastAPI"""

from typing import Optional
from functools import lru_cache

from models.mediapipe_analyzer import MediaPipeFaceAnalyzer
from services.face_detection_service import FaceDetectionService
from services.gemini_analysis_service import GeminiAnalysisService
from services.hybrid_recommender import HybridRecommender
from services.feedback_collector import FeedbackCollector
from services.retrain_queue import RetrainQueue
from core.logging import logger


# ========== Global Service Instances (Initialized at Startup) ==========
_mediapipe_analyzer: Optional[MediaPipeFaceAnalyzer] = None
_face_detection_service: Optional[FaceDetectionService] = None
_gemini_analysis_service: Optional[GeminiAnalysisService] = None
_hybrid_service: Optional[HybridRecommender] = None
_feedback_collector: Optional[FeedbackCollector] = None
_retrain_queue: Optional[RetrainQueue] = None


# ========== Initialization Functions (Called from main.py) ==========
def init_services(
    mediapipe_analyzer: Optional[MediaPipeFaceAnalyzer] = None,
    hybrid_service: Optional[HybridRecommender] = None,
    feedback_collector: Optional[FeedbackCollector] = None,
    retrain_queue: Optional[RetrainQueue] = None
) -> None:
    """
    Initialize global service instances

    Called from main.py startup event

    Args:
        mediapipe_analyzer: MediaPipe face analyzer
        hybrid_service: Hybrid recommendation service
        feedback_collector: Feedback collection service
        retrain_queue: Model retraining queue
    """
    global _mediapipe_analyzer, _face_detection_service, _gemini_analysis_service
    global _hybrid_service, _feedback_collector, _retrain_queue

    _mediapipe_analyzer = mediapipe_analyzer
    _hybrid_service = hybrid_service
    _feedback_collector = feedback_collector
    _retrain_queue = retrain_queue

    # Initialize services with dependencies
    _face_detection_service = FaceDetectionService(mediapipe_analyzer)
    _gemini_analysis_service = GeminiAnalysisService(max_retries=3)

    logger.info("✅ 의존성 주입 서비스 초기화 완료")


# ========== Dependency Providers (for FastAPI Depends) ==========
@lru_cache()
def get_face_detection_service() -> FaceDetectionService:
    """
    Get FaceDetectionService instance

    Returns:
        FaceDetectionService instance

    Raises:
        RuntimeError: If services not initialized
    """
    if _face_detection_service is None:
        raise RuntimeError(
            "FaceDetectionService not initialized. "
            "Call init_services() in main.py startup."
        )
    return _face_detection_service


@lru_cache()
def get_gemini_analysis_service() -> GeminiAnalysisService:
    """
    Get GeminiAnalysisService instance

    Returns:
        GeminiAnalysisService instance

    Raises:
        RuntimeError: If services not initialized
    """
    if _gemini_analysis_service is None:
        raise RuntimeError(
            "GeminiAnalysisService not initialized. "
            "Call init_services() in main.py startup."
        )
    return _gemini_analysis_service


@lru_cache()
def get_hybrid_service() -> HybridRecommender:
    """
    Get HybridRecommender instance

    Returns:
        HybridRecommender instance

    Raises:
        RuntimeError: If services not initialized
    """
    if _hybrid_service is None:
        raise RuntimeError(
            "HybridRecommender not initialized. "
            "Call init_services() in main.py startup."
        )
    return _hybrid_service


@lru_cache()
def get_feedback_collector() -> FeedbackCollector:
    """
    Get FeedbackCollector instance

    Returns:
        FeedbackCollector instance

    Raises:
        RuntimeError: If services not initialized
    """
    if _feedback_collector is None:
        raise RuntimeError(
            "FeedbackCollector not initialized. "
            "Call init_services() in main.py startup."
        )
    return _feedback_collector


@lru_cache()
def get_retrain_queue() -> RetrainQueue:
    """
    Get RetrainQueue instance

    Returns:
        RetrainQueue instance

    Raises:
        RuntimeError: If services not initialized
    """
    if _retrain_queue is None:
        raise RuntimeError(
            "RetrainQueue not initialized. "
            "Call init_services() in main.py startup."
        )
    return _retrain_queue
