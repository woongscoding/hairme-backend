"""Feedback submission endpoint (DynamoDB-only)

통계 조회는 관리자용 GET /api/admin/feedback-stats (routers/admin.py) 를 쓴다.
"""

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.logging import logger, log_structured
from database.dynamodb_connection import (
    save_feedback,
    get_analysis,
)
from api.dependencies import FeedbackRequest, FeedbackResponse

router = APIRouter()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@router.post("/feedback/submit", response_model=FeedbackResponse)
@limiter.limit("20/minute")  # 분당 20회 제한
async def submit_feedback(
    request: Request, feedback_data: FeedbackRequest
) -> FeedbackResponse:
    """
    User feedback submission endpoint (POST /api/feedback/submit)

    Args:
        request: Starlette Request (for SlowAPI rate limiting)
        feedback_data: FeedbackRequest model with:
            - analysis_id: str (analysis UUID)
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

    # style_index 화이트리스트 검증 (Pydantic ge=1 le=5, style_1~5 속성과 일치)
    VALID_STYLE_INDICES = {1, 2, 3, 4, 5}
    if feedback_data.style_index not in VALID_STYLE_INDICES:
        raise HTTPException(
            status_code=400, detail="유효하지 않은 스타일 인덱스입니다."
        )

    try:
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

        # 싫어요 사유 (선택) - 구버전 앱은 보내지 않는다
        dislike_reason = (
            feedback_data.dislike_reason.value
            if feedback_data.dislike_reason is not None
            else None
        )

        # Save feedback
        success = save_feedback(
            analysis_id=str(feedback_data.analysis_id),
            style_index=feedback_data.style_index,
            feedback=normalized_feedback,
            naver_clicked=feedback_data.naver_clicked,
            dislike_reason=dislike_reason,
        )

        if not success:
            raise HTTPException(status_code=500, detail="피드백 저장에 실패했습니다")

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
                "dislike_reason": dislike_reason,
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
