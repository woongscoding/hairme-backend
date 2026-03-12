"""Authentication and authorization utilities"""

import hmac
import logging
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from config.settings import settings

logger = logging.getLogger(__name__)

# API Key Header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_admin_api_key(api_key: str = Security(api_key_header)) -> str:
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="인증이 필요합니다.",
        )

    if not settings.ADMIN_API_KEY:
        logger.error("ADMIN_API_KEY가 서버에 설정되지 않았습니다")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="서비스를 일시적으로 사용할 수 없습니다.",
        )

    # 타이밍 공격 방지를 위한 상수 시간 비교
    if not hmac.compare_digest(api_key, settings.ADMIN_API_KEY):
        logger.warning("⚠️ Admin API 인증 실패 시도")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="인증에 실패했습니다.",
        )

    return api_key
