"""
ML 기반 헤어스타일 추천 서비스

MediaPipe 얼굴 분석 + ML 모델을 사용한 추천 제공

Author: HairMe ML Team
Date: 2025-11-08
Version: 2.0.0 (ML-only mode)
"""

import logging
from typing import List, Dict, Optional, Any
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.style_preprocessor import normalize_style_name

logger = logging.getLogger(__name__)


class MLRecommendationService:
    """ML 기반 헤어스타일 추천 서비스"""

    def __init__(self):
        """초기화"""
        # ML 추천기 로드 (Lazy import)
        try:
            from models.ml_recommender import get_ml_recommender
            self.ml_recommender = get_ml_recommender()
            self.ml_available = True
            logger.info("✅ ML 추천기 로드 성공")
        except Exception as e:
            logger.error(f"❌ ML 추천기 로드 실패: {str(e)}")
            self.ml_recommender = None
            self.ml_available = False
            raise

        # 추천 이유 생성기 로드 (Lazy import)
        try:
            from services.reason_generator import get_reason_generator
            self.reason_generator = get_reason_generator()
            logger.info("✅ 추천 이유 생성기 초기화 완료")
        except Exception as e:
            logger.warning(f"⚠️ 추천 이유 생성기 로드 실패: {str(e)}")
            self.reason_generator = None

    def _build_recommendations(
        self,
        ml_recommendations: List[Dict[str, Any]],
        face_shape: str,
        skin_tone: str
    ) -> List[Dict[str, Any]]:
        """
        ML 추천 결과를 응답 형식으로 변환

        Args:
            ml_recommendations: ML 추천 리스트
            face_shape: 얼굴형
            skin_tone: 피부톤

        Returns:
            추천 리스트
        """
        result = []
        seen_styles = set()

        for rec in ml_recommendations:
            hairstyle_id = rec.get("hairstyle_id")
            style_name = rec.get("hairstyle", "").strip()
            ml_score = rec.get("score", 0.0)

            if not style_name:
                continue

            # 띄어쓰기 정규화 적용 (중복 검사용)
            normalized_name = normalize_style_name(style_name)

            if normalized_name in seen_styles:
                continue

            # 템플릿 기반 이유 생성
            if self.reason_generator:
                try:
                    reason = self.reason_generator.generate_with_score(
                        face_shape, skin_tone, style_name, ml_score
                    )
                except Exception as e:
                    logger.warning(f"⚠️ 이유 생성 실패: {str(e)}")
                    reason = f"ML 모델 추천 (점수: {ml_score:.1f})"
            else:
                reason = f"ML 모델 추천 (점수: {ml_score:.1f})"

            result.append({
                "hairstyle_id": hairstyle_id,
                "style_name": style_name,
                "reason": reason,
                "source": "ml",
                "score": round(ml_score / 100.0, 2),  # 0-1 범위로 변환 (안드로이드 호환)
                "rank": len(result) + 1
            })

            seen_styles.add(normalized_name)

        logger.info(f"✅ ML 추천 결과: {len(result)}개")

        return result

    def recommend(
        self,
        image_data: bytes,
        face_shape: str,
        skin_tone: str,
        face_features: List[float] = None,
        skin_features: List[float] = None,
        gender: str = None
    ) -> Dict[str, Any]:
        """
        ML 기반 헤어스타일 추천

        Args:
            image_data: 이미지 바이트 (현재 사용 안함, 호환성 유지)
            face_shape: 얼굴형
            skin_tone: 피부톤
            face_features: MediaPipe 얼굴 측정값 [face_ratio, forehead_width, cheekbone_width, jaw_width, forehead_ratio, jaw_ratio] (6차원)
            skin_features: MediaPipe 피부 측정값 [ITA_value, hue_value] (2차원)
            gender: 성별 ("male", "female", "neutral")

        Returns:
            추천 결과 딕셔너리
        """
        if face_features is not None and skin_features is not None:
            logger.info(f"🎨 ML 추천 시작 (실제 측정값 사용): {face_shape} + {skin_tone}")
        else:
            logger.info(f"🎨 ML 추천 시작 (라벨 기반): {face_shape} + {skin_tone}")
            logger.warning("⚠️ 실제 측정값(face_features, skin_features)을 전달하는 것을 권장합니다.")

        # ML 추천 (Top-3, 성별 필터링 적용)
        ml_recommendations = []
        if self.ml_available and self.ml_recommender:
            try:
                ml_recommendations = self.ml_recommender.recommend_top_k(
                    face_shape=face_shape,
                    skin_tone=skin_tone,
                    k=3,
                    face_features=face_features,
                    skin_features=skin_features,
                    gender=gender
                )
                logger.info(f"✅ ML 추천 완료: {len(ml_recommendations)}개")
            except Exception as e:
                logger.error(f"❌ ML 추천 실패: {str(e)}")

        # 추천 결과 변환
        recommendations = self._build_recommendations(
            ml_recommendations,
            face_shape,
            skin_tone
        )

        # rank 재조정 (1, 2, 3)
        for idx, rec in enumerate(recommendations, 1):
            rec['rank'] = idx

        # 결과 구성
        result = {
            "analysis": {
                "face_shape": face_shape,
                "personal_color": skin_tone,
                "features": "ML 모델 기반 분석"
            },
            "recommendations": recommendations,
            "meta": {
                "total_count": len(recommendations),
                "ml_count": len(recommendations),
                "method": "ml"
            }
        }

        logger.info(f"✅ ML 추천 완료: 총 {len(recommendations)}개")

        return result


# ========== 하위 호환성을 위한 별칭 ==========
HybridRecommendationService = MLRecommendationService


# ========== 싱글톤 인스턴스 ==========
_ml_service_instance = None


def get_ml_recommendation_service() -> MLRecommendationService:
    """
    ML 추천 서비스 싱글톤 인스턴스 가져오기

    Returns:
        MLRecommendationService 인스턴스
    """
    global _ml_service_instance

    if _ml_service_instance is None:
        logger.info("🔧 ML 추천 서비스 초기화 중...")
        _ml_service_instance = MLRecommendationService()
        logger.info("✅ ML 추천 서비스 준비 완료")

    return _ml_service_instance


def create_hybrid_service(gemini_api_key: str = None) -> MLRecommendationService:
    """
    ML 추천 서비스 인스턴스 생성 (하위 호환성)

    Args:
        gemini_api_key: 사용 안함 (하위 호환성 유지)

    Returns:
        MLRecommendationService 인스턴스
    """
    return get_ml_recommendation_service()
