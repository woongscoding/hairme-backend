"""DynamoDB implementation of AnalysisRepository"""

import math
from typing import Optional, Dict, Any
from database.repository import AnalysisRepository
from database.dynamodb_connection import (
    save_analysis,
    get_analysis,
    save_feedback,
    get_feedback_stats,
)
from core.logging import logger


def _feature_version(mp_features: Any) -> str:
    """MediaPipeFaceFeatures.feature_version (문자열이 아니면 현재 측정 방식)"""
    from models.mediapipe_analyzer import FEATURE_VERSION

    value = getattr(mp_features, "feature_version", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return FEATURE_VERSION


def _optional_float(value: Any) -> Optional[float]:
    """유한한 숫자만 float 로 (None/NaN/목 객체는 None -> 저장하지 않음)"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


class DynamoDBAnalysisRepository(AnalysisRepository):
    """DynamoDB implementation of AnalysisRepository"""

    def save_analysis(
        self,
        image_hash: str,
        analysis_result: Dict[str, Any],
        processing_time: float,
        detection_method: str,
        mp_features: Optional[Any] = None,
        model_version: Optional[str] = None,
        gender: Optional[str] = None,
    ) -> Optional[str]:
        """
        Save analysis result to DynamoDB

        Returns:
            Analysis ID (UUID string) if successful, None otherwise
        """
        try:
            gemini_shape = analysis_result.get("analysis", {}).get("face_shape")
            recommendations = analysis_result.get("recommendations", [])

            # Calculate MediaPipe agreement
            mediapipe_agreement = None
            if mp_features:
                mediapipe_agreement = (
                    mp_features.face_shape in gemini_shape
                    or gemini_shape in mp_features.face_shape
                )

            # Build data dict for DynamoDB
            data = {
                "image_hash": image_hash,
                "face_shape": gemini_shape,
                "personal_color": analysis_result.get("analysis", {}).get(
                    "personal_color"
                ),
                "recommendations": recommendations,
                "recommended_styles": recommendations,
                "processing_time": processing_time,
                "detection_method": detection_method,
                "opencv_gemini_agreement": mediapipe_agreement,
                # 품질 전후 비교용 - 값이 없으면 dynamodb_connection 이 "unknown" 으로 채운다
                "model_version": model_version,
                # 추천 실험 배정 (없으면 dynamodb_connection 이 저장하지 않는다)
                "experiment_id": analysis_result.get("experiment_id"),
                "ab_variant": analysis_result.get("ab_variant"),
                # 반응률 보너스 실험 배정 (탐색 실험과 별도 필드, 없으면 저장하지 않는다)
                "engagement_arm": analysis_result.get("engagement_arm"),
                "engagement_version": analysis_result.get("engagement_version"),
                "engagement_applied": analysis_result.get("engagement_applied"),
            }

            # 성별은 요청에 있을 때만 저장 (분석 결과의 추정값이 있으면 그것을 우선)
            resolved_gender = gender or analysis_result.get("analysis", {}).get(
                "gender"
            )
            if resolved_gender:
                data["gender"] = resolved_gender

            # Add MediaPipe continuous features
            if mp_features:
                data["mediapipe_face_ratio"] = mp_features.face_ratio
                data["mediapipe_forehead_width"] = mp_features.forehead_width
                data["mediapipe_cheekbone_width"] = mp_features.cheekbone_width
                data["mediapipe_jaw_width"] = mp_features.jaw_width

                # Ratios (division by zero protection)
                if mp_features.cheekbone_width > 0:
                    data["mediapipe_forehead_ratio"] = (
                        mp_features.forehead_width / mp_features.cheekbone_width
                    )
                    data["mediapipe_jaw_ratio"] = (
                        mp_features.jaw_width / mp_features.cheekbone_width
                    )

                # Skin measurements
                data["mediapipe_ITA_value"] = mp_features.ITA_value
                data["mediapipe_hue_value"] = mp_features.hue_value

                # Metadata
                data["mediapipe_confidence"] = mp_features.confidence
                data["mediapipe_features_complete"] = True

                # 고개 각도 (2026-11-08~). 이 필드가 있는 행은 3D 거리로 잰 측정값이다
                data["mediapipe_head_yaw"] = getattr(mp_features, "head_yaw", None)
                data["mediapipe_head_pitch"] = getattr(mp_features, "head_pitch", None)

                # 측정 방식 표식: 트레이너가 2D(v1_2d)/3D(v2_3d) 샘플을 구분한다
                data["mediapipe_feature_version"] = _feature_version(mp_features)

                # 피부 영역 평균 L*/b* 원값 (쿨톤 임계값 재보정용, 분류에는 안 씀)
                data["mediapipe_L_star"] = _optional_float(
                    getattr(mp_features, "L_star", None)
                )
                data["mediapipe_b_star"] = _optional_float(
                    getattr(mp_features, "b_star", None)
                )

                logger.info(
                    f"✅ MediaPipe 연속형 변수 포함: ratio={mp_features.face_ratio:.2f}, ITA={mp_features.ITA_value:.1f}"
                )

            # Save to DynamoDB
            analysis_id = save_analysis(data)

            if analysis_id:
                logger.info(f"✅ DynamoDB 저장 성공 (ID: {analysis_id})")
                return analysis_id
            else:
                logger.error("❌ DynamoDB 저장 실패")
                return None

        except Exception as e:
            logger.error(f"❌ DynamoDB 저장 실패: {str(e)}")
            return None

    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis result from DynamoDB"""
        return get_analysis(analysis_id)

    def save_feedback(
        self, analysis_id: str, style_index: int, feedback: str, naver_clicked: bool
    ) -> bool:
        """Save user feedback to DynamoDB"""
        return save_feedback(analysis_id, style_index, feedback, naver_clicked)

    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get aggregated feedback statistics from DynamoDB"""
        return get_feedback_stats()
