"""회원 합성 결과 히스토리 + AI 생성 이미지 신고 엔드포인트"""

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.logging import log_structured, logger, mask_user_id
from core.jwt_auth import get_current_user_id, get_optional_user_id
from services.photo_storage_service import get_photo_storage_service
from services.usage_limit_service import validate_device_id

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

ReportReason = Literal[
    "sexual",  # 선정적
    "violent",  # 폭력적·혐오
    "face_altered",  # 얼굴이 다른 사람처럼 바뀜
    "offensive",  # 불쾌한 결과
    "other",
]


class ResultReportRequest(BaseModel):
    result_id: str = Field(
        ..., min_length=1, max_length=64, description="합성 응답의 result_id"
    )
    reason: ReportReason
    comment: Optional[str] = Field(None, max_length=500, description="추가 설명 (선택)")
    device_id: Optional[str] = Field(None, description="비회원 디바이스 ID (선택)")


@router.post("/results/report")
@limiter.limit("10/minute")
async def report_result(
    request: Request,
    body: ResultReportRequest,
    user_id: Optional[str] = Depends(get_optional_user_id),
):
    """
    AI 생성 이미지 신고 (회원/비회원 모두)

    결과 이미지는 저장하지 않고 신고 정보만 남긴다. result_id 는 합성 로그의
    cache_key 앞 16자와 같아서 CloudWatch 의 synthesis_success 줄과 맞춰 볼 수 있다.
    저장소 장애여도 구조화 로그에 같은 내용이 남으므로 사용자에게는 접수로 응답한다.
    """
    reporter = None
    if user_id:
        reporter = mask_user_id(user_id)
    elif body.device_id:
        try:
            reporter = mask_user_id(validate_device_id(body.device_id))
        except ValueError:
            reporter = None

    report = {
        "result_id": body.result_id,
        "reason": body.reason,
        "comment": body.comment,
        "reporter": reporter,
        "authenticated": user_id is not None,
    }
    report_key = await run_in_threadpool(
        get_photo_storage_service().save_report, report
    )
    # 자유 입력 comment 는 개인정보가 섞일 수 있어 S3 신고 객체에만 남기고 로그에는 길이만
    log_payload = {k: v for k, v in report.items() if k != "comment"}
    log_payload["comment_length"] = len(body.comment or "")
    log_structured("ai_image_report", {**log_payload, "stored": report_key is not None})

    return {
        "success": True,
        "message": "신고가 접수되었습니다. 검토 후 조치하겠습니다.",
    }


@router.get("/me/results")
async def get_my_results(
    limit: int = Query(20, ge=1, le=100, description="페이지당 결과 수"),
    continuation_token: Optional[str] = Query(
        None, description="이전 응답의 next_token (다음 페이지 조회)"
    ),
    user_id: str = Depends(get_current_user_id),
):
    """
    내 합성 결과 목록 (최신순, 마이페이지 재열람용)

    각 항목은 presigned URL로 반환되며 PHOTO_URL_EXPIRE_SECONDS 후 만료된다.
    next_token이 null이 아니면 continuation_token으로 넘겨 다음 페이지 조회.
    """
    service = get_photo_storage_service()
    try:
        # 동기 S3 호출이라 스레드풀로 - 이벤트 루프 차단 방지
        page = await run_in_threadpool(
            service.list_user_results, user_id, limit, continuation_token
        )
    except Exception:
        logger.error(f"❌ 결과 히스토리 조회 실패: user_id={user_id}", exc_info=True)
        raise HTTPException(status_code=500, detail="결과 조회에 실패했습니다.")

    # 결과 재열람 계측 (리텐션 지표). user_id 원문은 남기지 않는다.
    log_structured(
        "results_viewed",
        {
            "user_id": mask_user_id(user_id),
            "count": len(page["items"]),
            "limit": limit,
            "paged": continuation_token is not None,
            "has_next": page["next_token"] is not None,
        },
    )

    return {"results": page["items"], "next_token": page["next_token"]}
