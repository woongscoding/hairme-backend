"""Dependency injection providers for FastAPI (ML-only mode)

Each provider lazily creates a module-level singleton on first use.
The module globals are also read (without triggering a load) by main.py's
root and health endpoints to report whether a service is loaded yet.
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.mediapipe_analyzer import MediaPipeFaceAnalyzer
    from services.face_detection_service import FaceDetectionService
    from services.hybrid_recommender import MLRecommendationService

from core.logging import logger

# ========== Lazy Singleton Instances ==========
_mediapipe_analyzer: Optional["MediaPipeFaceAnalyzer"] = None
_face_detection_service: Optional["FaceDetectionService"] = None
_hybrid_service: Optional["MLRecommendationService"] = None


# ========== Dependency Providers (for FastAPI Depends) ==========
def get_mediapipe_analyzer() -> "MediaPipeFaceAnalyzer":
    """
    Get MediaPipeFaceAnalyzer instance (Lazy Initialization)
    """
    global _mediapipe_analyzer
    if _mediapipe_analyzer is None:
        logger.info("🐢 Lazy initializing MediaPipeFaceAnalyzer...")
        from models.mediapipe_analyzer import MediaPipeFaceAnalyzer

        _mediapipe_analyzer = MediaPipeFaceAnalyzer()
    return _mediapipe_analyzer


def get_face_detection_service() -> "FaceDetectionService":
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


def get_hybrid_service() -> "MLRecommendationService":
    """
    Get recommendation service instance (Lazy Initialization)
    """
    global _hybrid_service

    if _hybrid_service is None:
        logger.info("🐢 Lazy initializing MLRecommendationService...")
        try:
            from services.hybrid_recommender import get_ml_recommendation_service

            _hybrid_service = get_ml_recommendation_service()

            logger.info("✅ MLRecommendationService initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MLRecommendationService: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize ML service: {e}")
    return _hybrid_service
