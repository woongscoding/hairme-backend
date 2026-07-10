"""회원 합성 결과 히스토리 엔드포인트"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from core.logging import logger
from core.jwt_auth import get_current_user_id
from services.photo_storage_service import get_photo_storage_service

router = APIRouter()


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

    return {"results": page["items"], "next_token": page["next_token"]}
