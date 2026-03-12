"""Feedback submission and statistics endpoints"""

import os
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request
from typing import List, Dict, Any, Optional
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.logging import logger, log_structured
from database.models import AnalysisHistory
from database.connection import get_db_session
from api.dependencies import FeedbackRequest, FeedbackResponse

router = APIRouter()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@router.post("/feedback", response_model=FeedbackResponse)
@router.post("/feedback/submit", response_model=FeedbackResponse)
@limiter.limit("20/minute")  # 분당 20회 제한
async def submit_feedback(
    request: Request, feedback_data: FeedbackRequest
) -> FeedbackResponse:
    """
    User feedback submission endpoint (supports both /api/feedback and /api/feedback/submit)

    Supports both MySQL and DynamoDB backends based on USE_DYNAMODB env variable.

    Args:
        request: Starlette Request (for SlowAPI rate limiting)
        feedback_data: FeedbackRequest model with:
            - analysis_id: str or int (UUID for DynamoDB, integer for MySQL)
            - style_index: int (1, 2, or 3)
            - feedback: FeedbackEnum ('good' or 'bad')
            - naver_clicked: bool

    Returns:
        FeedbackResponse model

    Raises:
        HTTPException: 404 if analysis not found, 500 for database errors
    """
    # ========== Validate analysis_id is not None/null ==========
    if feedback_data.analysis_id is None:
        logger.error("❌ 피드백 요청 실패: analysis_id가 null입니다")
        raise HTTPException(
            status_code=400,
            detail="분석 ID가 제공되지 않았습니다. 앱을 재시작하고 다시 분석을 실행해주세요.",
        )

    # style_index 화이트리스트 검증
    VALID_STYLE_INDICES = {0, 1, 2, 3, 4}
    if feedback_data.style_index not in VALID_STYLE_INDICES:
        raise HTTPException(
            status_code=400, detail="유효하지 않은 스타일 인덱스입니다."
        )

    use_dynamodb = os.getenv("USE_DYNAMODB", "false").lower() == "true"

    # ========== DynamoDB Backend ==========
    if use_dynamodb:
        try:
            from database.dynamodb_connection import save_feedback, get_analysis

            # Validate analysis exists
            analysis = get_analysis(str(feedback_data.analysis_id))
            if not analysis:
                raise HTTPException(
                    status_code=404,
                    detail=f"분석 결과를 찾을 수 없습니다 (ID: {feedback_data.analysis_id})",
                )

            # Normalize feedback value: like->good, dislike->bad
            normalized_feedback = feedback_data.feedback.value
            if normalized_feedback == "like":
                normalized_feedback = "good"
            elif normalized_feedback == "dislike":
                normalized_feedback = "bad"

            # Save feedback
            success = save_feedback(
                analysis_id=str(feedback_data.analysis_id),
                style_index=feedback_data.style_index,
                feedback=normalized_feedback,
                naver_clicked=feedback_data.naver_clicked,
            )

            if not success:
                raise HTTPException(
                    status_code=500, detail="피드백 저장에 실패했습니다"
                )

            logger.info(
                f"✅ 피드백 저장 성공 (DynamoDB): analysis_id={feedback_data.analysis_id}, "
                f"style={feedback_data.style_index}, feedback={feedback_data.feedback.value}"
            )

            log_structured(
                "feedback_submitted",
                {
                    "backend": "dynamodb",
                    "analysis_id": str(feedback_data.analysis_id),
                    "style_index": feedback_data.style_index,
                    "feedback": feedback_data.feedback.value,
                    "naver_clicked": feedback_data.naver_clicked,
                },
            )

            return FeedbackResponse(
                success=True,
                message="피드백이 저장되었습니다",
                analysis_id=feedback_data.analysis_id,
                style_index=feedback_data.style_index,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ DynamoDB 피드백 저장 실패: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            )

    # ========== MySQL Backend (Original) ==========
    else:
        db = get_db_session()
        if not db:
            raise HTTPException(status_code=500, detail="데이터베이스 연결이 없습니다")

        try:
            record = (
                db.query(AnalysisHistory)
                .filter(AnalysisHistory.id == feedback_data.analysis_id)
                .first()
            )

            if not record:
                db.close()
                raise HTTPException(
                    status_code=404,
                    detail=f"분석 결과를 찾을 수 없습니다 (ID: {feedback_data.analysis_id})",
                )

            feedback_column = f"style_{feedback_data.style_index}_feedback"
            clicked_column = f"style_{feedback_data.style_index}_naver_clicked"

            # Normalize feedback value: like->good, dislike->bad
            normalized_feedback = feedback_data.feedback.value
            if normalized_feedback == "like":
                normalized_feedback = "good"
            elif normalized_feedback == "dislike":
                normalized_feedback = "bad"

            # Store normalized feedback value
            setattr(record, feedback_column, normalized_feedback)
            setattr(record, clicked_column, feedback_data.naver_clicked)
            record.feedback_at = datetime.utcnow()

            db.commit()

            logger.info(
                f"✅ 피드백 저장 성공 (MySQL): analysis_id={feedback_data.analysis_id}, "
                f"style={feedback_data.style_index}, feedback={feedback_data.feedback.value}"
            )

            log_structured(
                "feedback_submitted",
                {
                    "backend": "mysql",
                    "analysis_id": feedback_data.analysis_id,
                    "style_index": feedback_data.style_index,
                    "feedback": feedback_data.feedback.value,
                    "naver_clicked": feedback_data.naver_clicked,
                },
            )

            db.close()

            return FeedbackResponse(
                success=True,
                message="피드백이 저장되었습니다",
                analysis_id=feedback_data.analysis_id,
                style_index=feedback_data.style_index,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ MySQL 피드백 저장 실패: {str(e)}", exc_info=True)
            if db:
                db.close()
            raise HTTPException(
                status_code=500,
                detail="서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            )


@router.get("/stats/feedback")
@limiter.limit("30/minute")  # 분당 30회 제한 (통계 조회)
async def get_feedback_stats(request: Request) -> Dict[str, Any]:
    """
    Feedback statistics query API

    Supports both MySQL and DynamoDB backends based on USE_DYNAMODB env variable.

    Returns:
        - success: bool
        - total_analysis: Total analysis record count
        - total_feedback: Record count with feedback
        - recent_feedbacks: Latest 5 feedback data
        - like_counts: Like count by style (dict)
        - dislike_counts: Dislike count by style (dict)

    Raises:
        HTTPException: 500 for database errors
    """
    use_dynamodb = os.getenv("USE_DYNAMODB", "false").lower() == "true"

    # ========== DynamoDB Backend ==========
    if use_dynamodb:
        try:
            from database.dynamodb_connection import (
                get_feedback_stats as get_dynamodb_stats,
            )

            stats = get_dynamodb_stats()

            if not stats.get("success"):
                raise HTTPException(status_code=500, detail="통계 조회에 실패했습니다")

            logger.info(
                f"📊 통계 조회 (DynamoDB): 전체 {stats['total_analysis']}개, "
                f"피드백 {stats['total_feedback']}개"
            )

            return stats

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ DynamoDB 통계 조회 실패: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            )

    # ========== MySQL Backend (Original) ==========
    else:
        db = get_db_session()
        if not db:
            raise HTTPException(status_code=500, detail="데이터베이스 연결이 없습니다")

        try:
            # Total statistics
            total = db.query(AnalysisHistory).count()
            feedback_count = (
                db.query(AnalysisHistory)
                .filter(AnalysisHistory.feedback_at.isnot(None))
                .count()
            )

            # Latest 5 feedbacks
            recent = (
                db.query(AnalysisHistory)
                .filter(AnalysisHistory.feedback_at.isnot(None))
                .order_by(AnalysisHistory.id.desc())
                .limit(5)
                .all()
            )

            recent_data: List[Dict[str, Any]] = []
            for r in recent:
                entry = {
                    "id": r.id,
                    "face_shape": r.face_shape,
                    "personal_color": r.personal_color,
                    "style_1_feedback": r.style_1_feedback,
                    "style_2_feedback": r.style_2_feedback,
                    "style_3_feedback": r.style_3_feedback,
                    "style_1_naver_clicked": r.style_1_naver_clicked,
                    "style_2_naver_clicked": r.style_2_naver_clicked,
                    "style_3_naver_clicked": r.style_3_naver_clicked,
                    "feedback_at": r.feedback_at.isoformat() if r.feedback_at else None,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                # 트렌드 스타일 피드백 (컬럼이 존재하면 포함)
                for i in [4, 5]:
                    if hasattr(r, f"style_{i}_feedback"):
                        entry[f"style_{i}_feedback"] = getattr(
                            r, f"style_{i}_feedback", None
                        )
                    if hasattr(r, f"style_{i}_naver_clicked"):
                        entry[f"style_{i}_naver_clicked"] = getattr(
                            r, f"style_{i}_naver_clicked", False
                        )
                recent_data.append(entry)

            # Like/Dislike statistics
            like_counts = {
                "style_1": 0,
                "style_2": 0,
                "style_3": 0,
                "style_4": 0,
                "style_5": 0,
            }
            dislike_counts = {
                "style_1": 0,
                "style_2": 0,
                "style_3": 0,
                "style_4": 0,
                "style_5": 0,
            }

            all_feedback = (
                db.query(AnalysisHistory)
                .filter(AnalysisHistory.feedback_at.isnot(None))
                .all()
            )

            for record in all_feedback:
                for i in [1, 2, 3, 4, 5]:
                    fb_val = getattr(record, f"style_{i}_feedback", None)
                    if fb_val in ["like", "good"]:
                        like_counts[f"style_{i}"] += 1
                    elif fb_val in ["dislike", "bad"]:
                        dislike_counts[f"style_{i}"] += 1

            db.close()

            logger.info(
                f"📊 통계 조회 (MySQL): 전체 {total}개, 피드백 {feedback_count}개"
            )

            return {
                "success": True,
                "total_analysis": total,
                "total_feedback": feedback_count,
                "like_counts": like_counts,
                "dislike_counts": dislike_counts,
                "recent_feedbacks": recent_data,
            }

        except Exception as e:
            logger.error(f"❌ MySQL 통계 조회 실패: {str(e)}", exc_info=True)
            if db:
                db.close()
            raise HTTPException(
                status_code=500,
                detail="서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            )
