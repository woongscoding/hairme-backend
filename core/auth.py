"""Authentication and authorization utilities"""

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from config.settings import settings

# API Key Header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_admin_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify admin API key for protected endpoints

    Args:
        api_key: API key from X-API-Key header

    Returns:
        The validated API key

    Raises:
        HTTPException: 403 if API key is invalid or missing
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API Key is required. Please provide X-API-Key header."
        )

    if not settings.ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin API Key is not configured on server"
        )

    if api_key != settings.ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )

    return api_key
