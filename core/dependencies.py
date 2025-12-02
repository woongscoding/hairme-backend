"""Dependency injection providers for FastAPI (ML-only mode)"""

import os
from typing import Optional, TYPE_CHECKING, Union
from functools import lru_cache

if TYPE_CHECKING:
    from models.mediapipe_analyzer import MediaPipeFaceAnalyzer
    from services.face_detection_service import FaceDetectionService
    from services.hybrid_recommender import MLRecommendationService
    from models.onnx_recommender import ONNXHairstyleRecommender

from core.logging import logger

# Lambda 환경 감지
IS_LAMBDA = os.environ.get('AWS_LAMBDA_FUNCTION_NAME') is not None
USE_ONNX = os.environ.get('USE_ONNX', 'false').lower() == 'true' or IS_LAMBDA


# ========== Global Service Instances (Initialized at Startup) ==========
_mediapipe_analyzer: Optional['MediaPipeFaceAnalyzer'] = None
_face_detection_service: Optional['FaceDetectionService'] = None
_hybrid_service: Optional['MLRecommendationService'] = None
_onnx_recommender: Optional['ONNXHairstyleRecommender'] = None


# ========== Initialization Functions (Called from main.py) ==========
def init_services(
    mediapipe_analyzer: Optional['MediaPipeFaceAnalyzer'] = None,
    hybrid_service: Optional['MLRecommendationService'] = None
) -> None:
    """
    Initialize global service instances (ML-only mode)

    Called from main.py startup event

    Args:
        mediapipe_analyzer: MediaPipe face analyzer
        hybrid_service: ML recommendation service
    """
    global _mediapipe_analyzer, _face_detection_service
    global _hybrid_service

    _mediapipe_analyzer = mediapipe_analyzer
    _hybrid_service = hybrid_service

    # Initialize services with dependencies (lazy import)
    from services.face_detection_service import FaceDetectionService
    _face_detection_service = FaceDetectionService(mediapipe_analyzer)

    logger.info("✅ 의존성 주입 서비스 초기화 완료 (ML-only mode)")


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
def get_hybrid_service() -> Union['MLRecommendationService', 'ONNXRecommenderWrapper']:
    """
    Get recommendation service instance (Lazy Initialization)

    Lambda 환경에서는 ONNX 기반 경량 추천기를 사용하고,
    로컬 환경에서는 PyTorch 기반 MLRecommendationService를 사용합니다.
    """
    global _hybrid_service, _onnx_recommender

    # Lambda 환경 또는 USE_ONNX=true인 경우 ONNX 사용
    if USE_ONNX:
        if _onnx_recommender is None:
            logger.info("🚀 Lambda/ONNX 모드: ONNXHairstyleRecommender 초기화 중...")
            try:
                from models.onnx_recommender import get_onnx_recommender
                _onnx_recommender = get_onnx_recommender()

                # Update startup status
                try:
                    import main
                    main.startup_status["ml_service"] = True
                except Exception:
                    pass  # main 모듈이 없을 수 있음

                logger.info("✅ ONNX 추천기 초기화 완료 (콜드 스타트 최적화)")
            except Exception as e:
                logger.error(f"❌ ONNX 추천기 초기화 실패: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise RuntimeError(f"Failed to initialize ONNX recommender: {e}")

        # ONNX 추천기를 MLRecommendationService 인터페이스로 래핑
        return ONNXRecommenderWrapper(_onnx_recommender)

    # 로컬 환경: PyTorch 기반 MLRecommendationService 사용
    if _hybrid_service is None:
        logger.info("🐢 Lazy initializing MLRecommendationService (PyTorch)...")
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


class ONNXRecommenderWrapper:
    """
    ONNX 추천기를 MLRecommendationService 인터페이스로 래핑

    API 엔드포인트에서 동일한 인터페이스로 사용 가능
    """

    def __init__(self, onnx_recommender: 'ONNXHairstyleRecommender'):
        self.onnx_recommender = onnx_recommender
        self.ml_available = True
        self.reason_generator = None

        # Reason generator 로드 시도
        try:
            from services.reason_generator import get_reason_generator
            self.reason_generator = get_reason_generator()
        except Exception as e:
            logger.warning(f"⚠️ 추천 이유 생성기 로드 실패: {e}")

    def recommend(
        self,
        image_data: bytes,
        face_shape: str,
        skin_tone: str,
        face_features: list = None,
        skin_features: list = None,
        gender: str = None
    ) -> dict:
        """
        MLRecommendationService.recommend() 인터페이스 구현
        """
        from utils.style_preprocessor import normalize_style_name

        # ONNX 추천 실행
        ml_recommendations = self.onnx_recommender.recommend_top_k(
            face_shape=face_shape,
            skin_tone=skin_tone,
            k=3,
            face_features=face_features,
            skin_features=skin_features,
            gender=gender
        )

        # 응답 형식 변환 (MLRecommendationService와 동일)
        result = []
        seen_styles = set()

        for rec in ml_recommendations:
            hairstyle_id = rec.get("hairstyle_id")
            style_name = rec.get("hairstyle", "").strip()
            ml_score = rec.get("score", 0.0)

            if not style_name:
                continue

            normalized_name = normalize_style_name(style_name)
            if normalized_name in seen_styles:
                continue

            # 템플릿 기반 이유 생성
            if self.reason_generator:
                try:
                    reason = self.reason_generator.generate_with_score(
                        face_shape, skin_tone, style_name, ml_score
                    )
                except Exception:
                    reason = f"ML 모델 추천 (점수: {ml_score:.1f})"
            else:
                reason = f"ML 모델 추천 (점수: {ml_score:.1f})"

            result.append({
                "hairstyle_id": hairstyle_id,
                "style_name": style_name,
                "reason": reason,
                "source": "ml",
                "score": round(ml_score / 100.0, 2),
                "rank": len(result) + 1
            })

            seen_styles.add(normalized_name)

        # rank 재조정
        for idx, rec in enumerate(result, 1):
            rec['rank'] = idx

        return {
            "analysis": {
                "face_shape": face_shape,
                "personal_color": skin_tone,
                "features": "ONNX 모델 기반 분석"
            },
            "recommendations": result,
            "meta": {
                "total_count": len(result),
                "ml_count": len(result),
                "method": "onnx"
            }
        }


