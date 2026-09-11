"""Daily synthesis usage limit endpoints"""

from fastapi import APIRouter, HTTPException, Query

from config.settings import settings
from core.logging import logger
from services.usage_limit_service import get_usage_limit_service, validate_device_id

router = APIRouter()

LEGACY_DISABLED_MESSAGE = (
    "비로그인 무료 합성은 더 이상 제공되지 않습니다. 로그인 후 이용해주세요."
)


def _reject_if_legacy_flow_disabled() -> None:
    """레거시 device_id 흐름 킬 스위치 (settings.LEGACY_DEVICE_FLOW_ENABLED)

    410 Gone: 엔드포인트 자체가 폐지되었음을 알려 앱이 재시도하지 않도록 한다.
    """
    if not settings.LEGACY_DEVICE_FLOW_ENABLED:
        raise HTTPException(status_code=410, detail=LEGACY_DISABLED_MESSAGE)


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
    _reject_if_legacy_flow_disabled()

    try:
        validated_device_id = validate_device_id(device_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        service = get_usage_limit_service()
        usage = service.get_usage(validated_device_id)
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
    _reject_if_legacy_flow_disabled()

    try:
        validated_device_id = validate_device_id(device_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        service = get_usage_limit_service()
        result = service.get_usage(validated_device_id)
        return result

    except Exception as e:
        logger.error(f"Usage query failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail="사용량 조회 중 오류가 발생했습니다."
        )
