"""
BeautyMe Backend - Beauty Lambda (Lightweight)
Version: 1.0.0

Features:
- Hair Color Recommendation (GET /api/hair-color/{personal_color})
- Hair Color Synthesis (POST /api/hair-color/synthesize, Gemini)
- Usage (GET /api/usage)

NO PyTorch - Fast cold start (MediaPipe 도 import 하지 않는다. requirements-beauty.txt 정리는 별도)
"""

import os
from fastapi import Request

from config.settings import settings
from core.app_factory import (  # noqa: F401 - 테스트가 main_beauty._docs_kwargs 참조
    create_app,
    create_mangum_handler,
    docs_kwargs as _docs_kwargs,
    init_core_services,
)
from core.logging import logger

# Lambda type identifier
LAMBDA_TYPE = os.environ.get("LAMBDA_TYPE", "beauty")
IS_LAMBDA = os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None

# Import only beauty-related routers (NO ML, NO PyTorch)
from api.endpoints.hair_color import router as hair_color_router
from api.endpoints.usage import router as usage_router

# Lambda initialization flag
_lambda_initialized = False

# ========== Service Startup Status Tracking ==========
# "gemini" = GEMINI_API_KEY 설정됨 (google-genai Client 는 호출 시점에 생성)
startup_status = {"gemini": False}


# ========== FastAPI App ==========
# Sentry·문서 노출·한국어 429·TrustedHost/CORS/업로드 제한/보안 헤더는 공통 factory 담당
app = create_app(
    title="BeautyMe Beauty API",
    description="Hair Color Recommendation + Synthesis (Lightweight)",
    version="1.0.0",
)


# ========== Core Service Initialization (shared) ==========
def _init_core_services(strict_db: bool = False) -> None:
    """Gemini 키 확인 / DynamoDB 초기화 (startup 이벤트와 Lambda 초기화 공용)"""
    init_core_services(startup_status, strict_db=strict_db)


# ========== Lambda Initialization Helper ==========
def ensure_lambda_initialization():
    """Lambda 컨테이너당 1회만 핵심 서비스를 초기화한다 (lifespan="off")"""
    global _lambda_initialized

    if _lambda_initialized:
        return

    if not IS_LAMBDA:
        return

    logger.info("Lambda cold start - initializing beauty services...")

    _init_core_services(strict_db=False)

    _lambda_initialized = True
    logger.info("Beauty Lambda initialization complete")


# ========== Lambda Initialization Middleware ==========
@app.middleware("http")
async def lambda_init_middleware(request: Request, call_next):
    """Ensure Lambda is initialized before processing requests"""
    if IS_LAMBDA and not _lambda_initialized:
        ensure_lambda_initialization()
    return await call_next(request)


# ========== Register Routers (Beauty Only) ==========
app.include_router(hair_color_router, prefix="/api", tags=["hair_color"])
app.include_router(usage_router, prefix="/api", tags=["usage"])


# ========== Startup Event ==========
@app.on_event("startup")
async def startup_event():
    """로컬 uvicorn 실행 시 핵심 서비스를 초기화한다 (Lambda 는 lifespan="off")"""
    logger.info("Beauty Lambda starting...")

    _init_core_services(strict_db=True)

    logger.info("Beauty Lambda startup complete")


# ========== Root Endpoint ==========
@app.get("/")
async def root():
    """Root endpoint with service status"""
    return {
        "name": "BeautyMe Beauty API",
        "message": "Hair Color Recommendation + Synthesis",
        "version": "1.0.0",
        "lambda_type": LAMBDA_TYPE,
        "status": "running",
        "features": {
            "hair_color_recommendation": "enabled",
            "hair_color_synthesis": (
                "enabled" if settings.GEMINI_API_KEY else "disabled"
            ),
        },
        "endpoints": {
            "hair_color": "/api/hair-color/{type}",
            "hair_color_synthesis": "/api/hair-color/synthesize",
        },
    }


# ========== Health Check Endpoint ==========
@app.get("/api/health")
async def health_check():
    """
    Health check for Beauty Lambda (배포 게이트: status == "healthy").

    필수 서비스(gemini)가 준비되면 healthy 다.
    """
    return {
        "status": "healthy" if startup_status["gemini"] else "degraded",
        "version": "1.0.0",
        "lambda_type": LAMBDA_TYPE,
        "environment": settings.ENVIRONMENT,
        "services": {
            "required": {"gemini": startup_status["gemini"]},
        },
    }


# ========== Lambda Handler ==========
# lifespan="off" (core.app_factory 참고). Lambda 초기화는 lambda_init_middleware 담당.
handler = create_mangum_handler(app, "Beauty Lambda")


# ========== Main Entry Point ==========
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
