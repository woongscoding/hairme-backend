"""Daily synthesis usage limit endpoints"""

from fastapi import APIRouter, HTTPException, Query

from core.logging import logger
from services.usage_limit_service import get_usage_limit_service

router = APIRouter()


@router.get("/usage")
async def get_usage(
    device_id: str = Query(..., description="디바이스 고유 ID"),
):
    """
    남은 합성 횟수 조회 API

    Args:
        device_id: 디바이스 고유 식별자

    Returns:
        { "daily_limit": 3, "used": 2, "remaining": 1 }
    """
    if not device_id or not device_id.strip():
        raise HTTPException(status_code=400, detail="device_id는 필수입니다.")

    try:
        service = get_usage_limit_service()
        usage = service.get_usage(device_id.strip())
        return usage

    except Exception as e:
        logger.error(f"Usage query failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail="사용량 조회 중 오류가 발생했습니다."
        )


@router.post("/usage/consume")
async def consume_usage(
    device_id: str = Query(..., description="디바이스 고유 ID"),
):
    """
    합성 횟수 1회 소비 API

    클라이언트에서 합성(헤어스타일/염색색) 성공 후 호출합니다.

    Args:
        device_id: 디바이스 고유 식별자

    Returns:
        { "daily_limit": 3, "used": 2, "remaining": 1 }
    """
    if not device_id or not device_id.strip():
        raise HTTPException(status_code=400, detail="device_id는 필수입니다.")

    try:
        service = get_usage_limit_service()
        result = service.increment_usage(device_id.strip())
        return result

    except Exception as e:
        logger.error(f"Usage consume failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail="사용량 소비 중 오류가 발생했습니다."
        )
