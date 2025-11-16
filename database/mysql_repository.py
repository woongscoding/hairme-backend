"""MySQL implementation of AnalysisRepository"""

from typing import Optional, Dict, Any
from database.repository import AnalysisRepository
from database.models import AnalysisHistory
from database.connection import get_db_session
from core.logging import logger, log_structured


class MySQLAnalysisRepository(AnalysisRepository):
    """MySQL implementation of AnalysisRepository"""

    def save_analysis(
        self,
        image_hash: str,
        analysis_result: Dict[str, Any],
        processing_time: float,
        detection_method: str,
        mp_features: Optional[Any] = None
    ) -> Optional[int]:
        """
        Save analysis result to MySQL

        Returns:
            Record ID (int) if successful, None otherwise
        """
        db = get_db_session()
        if not db:
            logger.warning("⚠️ 데이터베이스 연결이 없어 저장을 생략합니다.")
            return None

        try:
            gemini_shape = analysis_result.get("analysis", {}).get("face_shape")

            # Calculate MediaPipe agreement
            mediapipe_agreement = None
            if mp_features:
                mediapipe_agreement = (
                    mp_features.face_shape in gemini_shape or
                    gemini_shape in mp_features.face_shape
                )

            recommendations = analysis_result.get("recommendations", [])

            history = AnalysisHistory(
                image_hash=image_hash,
                face_shape=gemini_shape,
                personal_color=analysis_result.get("analysis", {}).get("personal_color"),
                recommendations=recommendations,
                recommended_styles=recommendations,
                processing_time=processing_time,
                detection_method=detection_method,
                opencv_gemini_agreement=mediapipe_agreement,
            )

            # MediaPipe continuous features
            if mp_features:
                # Face measurements
                history.mediapipe_face_ratio = mp_features.face_ratio
                history.mediapipe_forehead_width = mp_features.forehead_width
                history.mediapipe_cheekbone_width = mp_features.cheekbone_width
                history.mediapipe_jaw_width = mp_features.jaw_width

                # Ratios (division by zero protection)
                if mp_features.cheekbone_width > 0:
                    history.mediapipe_forehead_ratio = mp_features.forehead_width / mp_features.cheekbone_width
                    history.mediapipe_jaw_ratio = mp_features.jaw_width / mp_features.cheekbone_width

                # Skin measurements
                history.mediapipe_ITA_value = mp_features.ITA_value
                history.mediapipe_hue_value = mp_features.hue_value

                # Metadata
                history.mediapipe_confidence = mp_features.confidence
                history.mediapipe_features_complete = True

                logger.info(f"✅ MediaPipe 연속형 변수 저장: ratio={mp_features.face_ratio:.2f}, ITA={mp_features.ITA_value:.1f}")

            db.add(history)
            db.commit()
            db.refresh(history)

            logger.info(f"✅ MySQL 저장 성공 (ID: {history.id})")
            log_structured("database_saved", {
                "backend": "mysql",
                "record_id": history.id,
                "mediapipe_enabled": mp_features is not None,
                "mediapipe_agreement": mediapipe_agreement,
                "recommendations_count": len(recommendations)
            })

            db.close()
            return history.id

        except Exception as e:
            logger.error(f"❌ MySQL 저장 실패: {str(e)}")
            db.close()
            return None

    def get_analysis(self, analysis_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve analysis result from MySQL"""
        db = get_db_session()
        if not db:
            return None

        try:
            record = db.query(AnalysisHistory).filter(
                AnalysisHistory.id == analysis_id
            ).first()

            if record:
                result = {
                    'id': record.id,
                    'image_hash': record.image_hash,
                    'face_shape': record.face_shape,
                    'personal_color': record.personal_color,
                    'recommendations': record.recommendations,
                    'processing_time': record.processing_time,
                    'detection_method': record.detection_method,
                    'created_at': record.created_at.isoformat() if record.created_at else None
                }
                db.close()
                return result
            else:
                db.close()
                return None

        except Exception as e:
            logger.error(f"❌ MySQL 조회 실패: {str(e)}")
            db.close()
            return None

    def save_feedback(
        self,
        analysis_id: int,
        style_index: int,
        feedback: str,
        naver_clicked: bool
    ) -> bool:
        """Save user feedback to MySQL"""
        db = get_db_session()
        if not db:
            return False

        try:
            record = db.query(AnalysisHistory).filter(
                AnalysisHistory.id == analysis_id
            ).first()

            if not record:
                logger.error(f"❌ Analysis not found: {analysis_id}")
                db.close()
                return False

            # Update feedback fields based on style_index
            if style_index == 1:
                record.style_1_feedback = feedback
                record.style_1_naver_clicked = naver_clicked
            elif style_index == 2:
                record.style_2_feedback = feedback
                record.style_2_naver_clicked = naver_clicked
            elif style_index == 3:
                record.style_3_feedback = feedback
                record.style_3_naver_clicked = naver_clicked

            db.commit()
            logger.info(f"✅ MySQL 피드백 저장 완료: ID={analysis_id}, style={style_index}")
            db.close()
            return True

        except Exception as e:
            logger.error(f"❌ MySQL 피드백 저장 실패: {str(e)}")
            db.close()
            return False

    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get aggregated feedback statistics from MySQL"""
        db = get_db_session()
        if not db:
            return {
                'success': False,
                'total_analysis': 0,
                'total_feedback': 0,
                'like_counts': {'style_1': 0, 'style_2': 0, 'style_3': 0},
                'dislike_counts': {'style_1': 0, 'style_2': 0, 'style_3': 0},
                'recent_feedbacks': []
            }

        try:
            # Total analysis count
            total_analysis = db.query(AnalysisHistory).count()

            # Feedback counts
            records = db.query(AnalysisHistory).all()

            like_counts = {'style_1': 0, 'style_2': 0, 'style_3': 0}
            dislike_counts = {'style_1': 0, 'style_2': 0, 'style_3': 0}
            total_feedback = 0

            for record in records:
                for i in [1, 2, 3]:
                    feedback_field = getattr(record, f'style_{i}_feedback', None)
                    if feedback_field in ['good', 'like']:
                        like_counts[f'style_{i}'] += 1
                        total_feedback += 1
                    elif feedback_field in ['bad', 'dislike']:
                        dislike_counts[f'style_{i}'] += 1
                        total_feedback += 1

            # Recent feedbacks
            recent = db.query(AnalysisHistory).filter(
                AnalysisHistory.feedback_at.isnot(None)
            ).order_by(AnalysisHistory.feedback_at.desc()).limit(5).all()

            recent_data = []
            for record in recent:
                recent_data.append({
                    'id': record.id,
                    'face_shape': record.face_shape,
                    'personal_color': record.personal_color,
                    'style_1_feedback': record.style_1_feedback,
                    'style_2_feedback': record.style_2_feedback,
                    'style_3_feedback': record.style_3_feedback,
                    'style_1_naver_clicked': record.style_1_naver_clicked,
                    'style_2_naver_clicked': record.style_2_naver_clicked,
                    'style_3_naver_clicked': record.style_3_naver_clicked,
                    'feedback_at': record.feedback_at.isoformat() if record.feedback_at else None,
                    'created_at': record.created_at.isoformat() if record.created_at else None
                })

            db.close()

            return {
                'success': True,
                'total_analysis': total_analysis,
                'total_feedback': total_feedback,
                'like_counts': like_counts,
                'dislike_counts': dislike_counts,
                'recent_feedbacks': recent_data
            }

        except Exception as e:
            logger.error(f"❌ MySQL 통계 조회 실패: {str(e)}")
            db.close()
            return {
                'success': False,
                'total_analysis': 0,
                'total_feedback': 0,
                'like_counts': {'style_1': 0, 'style_2': 0, 'style_3': 0},
                'dislike_counts': {'style_1': 0, 'style_2': 0, 'style_3': 0},
                'recent_feedbacks': []
            }
