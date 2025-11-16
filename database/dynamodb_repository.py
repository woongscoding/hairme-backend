"""DynamoDB implementation of AnalysisRepository"""

from typing import Optional, Dict, Any
from database.repository import AnalysisRepository
from database.dynamodb_connection import (
    save_analysis,
    get_analysis,
    save_feedback,
    get_feedback_stats
)
from core.logging import logger


class DynamoDBAnalysisRepository(AnalysisRepository):
    """DynamoDB implementation of AnalysisRepository"""

    def save_analysis(
        self,
        image_hash: str,
        analysis_result: Dict[str, Any],
        processing_time: float,
        detection_method: str,
        mp_features: Optional[Any] = None
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
                    mp_features.face_shape in gemini_shape or
                    gemini_shape in mp_features.face_shape
                )

            # Build data dict for DynamoDB
            data = {
                'image_hash': image_hash,
                'face_shape': gemini_shape,
                'personal_color': analysis_result.get("analysis", {}).get("personal_color"),
                'recommendations': recommendations,
                'recommended_styles': recommendations,
                'processing_time': processing_time,
                'detection_method': detection_method,
                'opencv_gemini_agreement': mediapipe_agreement,
            }

            # Add MediaPipe continuous features
            if mp_features:
                data['mediapipe_face_ratio'] = mp_features.face_ratio
                data['mediapipe_forehead_width'] = mp_features.forehead_width
                data['mediapipe_cheekbone_width'] = mp_features.cheekbone_width
                data['mediapipe_jaw_width'] = mp_features.jaw_width

                # Ratios (division by zero protection)
                if mp_features.cheekbone_width > 0:
                    data['mediapipe_forehead_ratio'] = mp_features.forehead_width / mp_features.cheekbone_width
                    data['mediapipe_jaw_ratio'] = mp_features.jaw_width / mp_features.cheekbone_width

                # Skin measurements
                data['mediapipe_ITA_value'] = mp_features.ITA_value
                data['mediapipe_hue_value'] = mp_features.hue_value

                # Metadata
                data['mediapipe_confidence'] = mp_features.confidence
                data['mediapipe_features_complete'] = True

                logger.info(f"✅ MediaPipe 연속형 변수 포함: ratio={mp_features.face_ratio:.2f}, ITA={mp_features.ITA_value:.1f}")

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
        self,
        analysis_id: str,
        style_index: int,
        feedback: str,
        naver_clicked: bool
    ) -> bool:
        """Save user feedback to DynamoDB"""
        return save_feedback(analysis_id, style_index, feedback, naver_clicked)

    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get aggregated feedback statistics from DynamoDB"""
        return get_feedback_stats()
