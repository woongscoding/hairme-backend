"""Redis caching utilities"""

import json
import hashlib
from typing import Optional, Dict, Any
import redis

from config.settings import settings
from core.logging import logger, log_structured


# Global Redis client
redis_client: Optional[redis.Redis] = None


def init_redis() -> bool:
    """
    Initialize Redis connection

    Returns:
        bool: True if successful, False otherwise
    """
    global redis_client

    if not settings.REDIS_URL:
        logger.warning("⚠️ REDIS_URL 환경변수가 설정되지 않았습니다.")
        return False

    try:
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info(f"✅ Redis 연결 성공: {settings.REDIS_URL}")
        return True

    except Exception as e:
        logger.error(f"❌ Redis 연결 실패: {str(e)}")
        redis_client = None
        return False


def calculate_image_hash(image_data: bytes) -> str:
    """
    Calculate SHA256 hash of image data (used as caching key)

    Args:
        image_data: Image binary data

    Returns:
        SHA256 hash string
    """
    return hashlib.sha256(image_data).hexdigest()


def get_cached_result(image_hash: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached analysis result from Redis

    Args:
        image_hash: SHA256 hash of the image

    Returns:
        Cached result dict or None if not found
    """
    if not redis_client:
        return None

    try:
        cached = redis_client.get(f"analysis:{image_hash}")
        if cached:
            log_structured("cache_hit", {"image_hash": image_hash[:16]})
            return json.loads(cached)
        return None

    except Exception as e:
        logger.error(f"Redis 조회 중 오류: {str(e)}")
        return None


def save_to_cache(image_hash: str, result: Dict[str, Any]) -> bool:
    """
    Save analysis result to Redis

    Args:
        image_hash: SHA256 hash of the image
        result: Analysis result dict

    Returns:
        bool: True if successful, False otherwise
    """
    if not redis_client:
        return False

    try:
        redis_client.setex(
            f"analysis:{image_hash}",
            settings.CACHE_TTL,
            json.dumps(result, ensure_ascii=False)
        )
        return True

    except Exception as e:
        logger.error(f"Redis 저장 중 오류: {str(e)}")
        return False
