"""Dependency injection providers for FastAPI"""

from typing import Optional, TYPE_CHECKING
from functools import lru_cache

if TYPE_CHECKING:
    from models.mediapipe_analyzer import MediaPipeFaceAnalyzer
    from services.face_detection_service import FaceDetectionService
    from services.gemini_analysis_service import GeminiAnalysisService
    from services.hybrid_recommender import MLRecommendationService
    from services.feedback_collector import FeedbackCollector
    from services.retrain_queue import RetrainQueue
from core.logging import logger


# ========== Global Service Instances (Initialized at Startup) ==========
_mediapipe_analyzer: Optional['MediaPipeFaceAnalyzer'] = None
_face_detection_service: Optional['FaceDetectionService'] = None
_gemini_analysis_service: Optional['GeminiAnalysisService'] = None
_hybrid_service: Optional['MLRecommendationService'] = None
_feedback_collector: Optional['FeedbackCollector'] = None
_retrain_queue: Optional['RetrainQueue'] = None


# ========== Initialization Functions (Called from main.py) ==========
def init_services(
    mediapipe_analyzer: Optional['MediaPipeFaceAnalyzer'] = None,
    hybrid_service: Optional['MLRecommendationService'] = None,
    feedback_collector: Optional['FeedbackCollector'] = None,
    retrain_queue: Optional['RetrainQueue'] = None
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

    # Initialize services with dependencies (lazy import)
    from services.face_detection_service import FaceDetectionService
    from services.gemini_analysis_service import GeminiAnalysisService
    _face_detection_service = FaceDetectionService(mediapipe_analyzer)
    _gemini_analysis_service = GeminiAnalysisService(max_retries=3)

    logger.info("✅ 의존성 주입 서비스 초기화 완료")


# ========== Dependency Providers (for FastAPI Depends) ==========
@lru_cache()
def get_mediapipe_analyzer() -> 'MediaPipeFaceAnalyzer':
    """
    Get MediaPipeFaceAnalyzer instance (Lazy Initialization)
    """
    global _mediapipe_analyzer
    if _mediapipe_analyzer is None:
        logger.info("🐢 Lazy initializing MediaPipeFaceAnalyzer...")
        from models.mediapipe_analyzer import MediaPipeFaceAnalyzer
        _mediapipe_analyzer = MediaPipeFaceAnalyzer()
        # Update startup status
        import main
        main.startup_status["mediapipe"] = True
    return _mediapipe_analyzer


@lru_cache()
def get_face_detection_service() -> 'FaceDetectionService':
    """
    Get FaceDetectionService instance (Lazy Initialization)
    """
    global _face_detection_service
    if _face_detection_service is None:
        logger.info("🐢 Lazy initializing FaceDetectionService...")
        # Ensure MediaPipe is initialized
        mp_analyzer = get_mediapipe_analyzer()
        from services.face_detection_service import FaceDetectionService
        _face_detection_service = FaceDetectionService(mp_analyzer)
    return _face_detection_service


@lru_cache()
def get_gemini_analysis_service() -> 'GeminiAnalysisService':
    """
    Get GeminiAnalysisService instance (Lazy Initialization)
    """
    global _gemini_analysis_service
    if _gemini_analysis_service is None:
        logger.info("🐢 Lazy initializing GeminiAnalysisService...")
        from services.gemini_analysis_service import GeminiAnalysisService
        _gemini_analysis_service = GeminiAnalysisService(max_retries=3)
    return _gemini_analysis_service


@lru_cache()
def get_hybrid_service() -> 'MLRecommendationService':
    """
    Get MLRecommendationService instance (Lazy Initialization)
    """
    global _hybrid_service
    if _hybrid_service is None:
        logger.info("🐢 Lazy initializing MLRecommendationService...")
        try:
            from services.hybrid_recommender import get_ml_recommendation_service

            _hybrid_service = get_ml_recommendation_service()

            # Update startup status
            import main
            main.startup_status["ml_service"] = True

            logger.info("✅ MLRecommendationService initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MLRecommendationService: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize ML service: {e}")
    return _hybrid_service


@lru_cache()
def get_feedback_collector() -> Optional['FeedbackCollector']:
    """
    Get FeedbackCollector instance

    Returns:
        FeedbackCollector instance or None if not available

    Note:
        FeedbackCollector is optional. Returns None if initialization failed.
    """
    return _feedback_collector


@lru_cache()
def get_retrain_queue() -> 'RetrainQueue':
    """
    Get RetrainQueue instance (Lazy Initialization)
    """
    global _retrain_queue
    if _retrain_queue is None:
        logger.info("🐢 Lazy initializing RetrainQueue...")
        from services.retrain_queue import get_retrain_queue as _create_retrain_queue
        _retrain_queue = _create_retrain_queue()
    return _retrain_queue
