"""
Enhanced Health Check Service

Provides comprehensive health checks for all critical services:
- DynamoDB connectivity
- Gemini API availability
- System metrics (CPU, memory)
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
import psutil

logger = logging.getLogger(__name__)

# deep 헬스 체크의 Gemini 메타데이터 조회 HTTP 타임아웃 (초)
GEMINI_HEALTH_TIMEOUT_SECONDS = 10


class HealthCheckService:
    """Comprehensive health check service"""

    def __init__(self):
        """Initialize health check service"""
        self.last_check_time: Optional[float] = None
        self.cache_ttl_seconds = 30  # Cache results for 30 seconds

    async def check_dynamodb(self) -> Dict[str, Any]:
        """
        Check DynamoDB connection and table accessibility

        Returns:
            Dict with status, latency, and error details
        """
        from config.settings import settings

        if not settings.USE_DYNAMODB:
            return {
                "status": "skipped",
                "message": "DynamoDB not enabled (USE_DYNAMODB=false)",
                "latency_ms": 0,
            }

        try:
            import boto3
            from botocore.exceptions import ClientError

            start_time = time.time()

            # Create DynamoDB resource
            dynamodb = boto3.resource("dynamodb", region_name=settings.AWS_REGION)
            table = dynamodb.Table(settings.DYNAMODB_TABLE_NAME)

            # Lightweight operation - just check table status
            table_description = table.table_status

            latency_ms = round((time.time() - start_time) * 1000, 2)

            logger.info(f"✅ DynamoDB health check passed ({latency_ms}ms)")

            return {
                "status": "healthy",
                "latency_ms": latency_ms,
            }

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            logger.error(f"❌ DynamoDB health check failed: {error_code}")

            return {
                "status": "unhealthy",
                "latency_ms": 0,
            }

        except Exception as e:
            logger.error(f"❌ DynamoDB health check error: {str(e)}")

            return {
                "status": "unhealthy",
                "latency_ms": 0,
            }

    async def check_gemini_api(self) -> Dict[str, Any]:
        """
        Check Gemini API availability with a metadata lookup (no generation)

        google-genai ``models.get`` 로 이미지 생성 모델의 메타데이터만 조회한다.
        키 인증과 모델 접근 권한을 확인하며 토큰 과금이 없다.

        Returns:
            Dict with status and latency
        """
        from config.settings import settings

        try:
            from google import genai
            from google.genai import types

            start_time = time.time()

            client = genai.Client(
                api_key=settings.GEMINI_API_KEY,
                http_options=types.HttpOptions(
                    timeout=GEMINI_HEALTH_TIMEOUT_SECONDS * 1000
                ),
            )
            client.models.get(model=settings.GEMINI_IMAGE_MODEL)

            latency_ms = round((time.time() - start_time) * 1000, 2)

            logger.info(f"✅ Gemini API health check passed ({latency_ms}ms)")

            return {
                "status": "healthy",
                "latency_ms": latency_ms,
            }

        except Exception as e:
            logger.error(f"❌ Gemini API health check failed: {str(e)}")

            return {
                "status": "unhealthy",
                "latency_ms": 0,
            }

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get system resource usage metrics

        Returns:
            Dict with CPU, memory, and disk usage
        """
        try:
            # Get CPU usage (0.1 second interval)
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Get memory usage
            memory = psutil.virtual_memory()

            # Get disk usage (if available)
            try:
                disk = psutil.disk_usage("/")
                disk_info = {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent": disk.percent,
                }
            except (OSError, PermissionError) as e:
                # Disk not accessible or permission denied
                logger.warning(f"⚠️ Disk usage unavailable: {str(e)}")
                disk_info = {"status": "unavailable"}
            except Exception as e:
                # Unexpected error
                logger.error(f"❌ Disk usage error: {type(e).__name__}: {str(e)}")
                disk_info = {"status": "error"}

            return {
                "cpu": {"percent": cpu_percent, "count": psutil.cpu_count()},
                "memory": {
                    "total_mb": round(memory.total / (1024**2), 2),
                    "available_mb": round(memory.available / (1024**2), 2),
                    "used_mb": round(memory.used / (1024**2), 2),
                    "percent": memory.percent,
                },
                "disk": disk_info,
            }

        except Exception as e:
            logger.error(f"❌ System metrics error: {str(e)}")

            return {"status": "unavailable"}

    async def comprehensive_health_check(
        self, include_expensive_checks: bool = False
    ) -> Dict[str, Any]:
        """
        Run comprehensive health check

        Args:
            include_expensive_checks: If True, runs Gemini API check (models.get, ~1s)

        Returns:
            Dict with all health check results
        """
        start_time = time.time()

        # Basic status
        health_result = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
        }

        # 1. System metrics (fast, always run)
        health_result["checks"]["system"] = self.get_system_metrics()

        # 2. DynamoDB check (moderate cost, always run)
        dynamodb_result = await self.check_dynamodb()
        health_result["checks"]["dynamodb"] = dynamodb_result

        if dynamodb_result["status"] == "unhealthy":
            health_result["status"] = "degraded"

        # 3. Gemini API check (expensive, optional)
        if include_expensive_checks:
            gemini_result = await self.check_gemini_api()
            health_result["checks"]["gemini_api"] = gemini_result

            if gemini_result["status"] == "unhealthy":
                health_result["status"] = "degraded"
        else:
            health_result["checks"]["gemini_api"] = {
                "status": "skipped",
                "message": "Use ?deep=true for full check",
            }

        # Calculate total check time
        total_time_ms = round((time.time() - start_time) * 1000, 2)
        health_result["check_duration_ms"] = total_time_ms

        return health_result


# Singleton instance
_health_check_service = None


def get_health_check_service() -> HealthCheckService:
    """Get singleton health check service instance"""
    global _health_check_service

    if _health_check_service is None:
        _health_check_service = HealthCheckService()

    return _health_check_service
