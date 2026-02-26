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
    합성 횟수 소비 API (Deprecated)

    사용량 증가는 합성 API 호출 시 서버에서 자동 처리됩니다.
    이 엔드포인트는 하위 호환성을 위해 유지되며, 현재 사용량만 반환합니다.

    Args:
        device_id: 디바이스 고유 식별자

    Returns:
        { "daily_limit": 3, "used": 2, "remaining": 1 }
    """
    if not device_id or not device_id.strip():
        raise HTTPException(status_code=400, detail="device_id는 필수입니다.")

    try:
        service = get_usage_limit_service()
        result = service.get_usage(device_id.strip())
        return result

    except Exception as e:
        logger.error(f"Usage query failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail="사용량 조회 중 오류가 발생했습니다."
        )
