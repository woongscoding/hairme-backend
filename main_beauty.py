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
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config.settings import settings
from core.logging import logger, log_structured
from core.monitoring import init_sentry
from core.rate_limit import rate_limit_exceeded_handler

# Lambda type identifier
LAMBDA_TYPE = os.environ.get("LAMBDA_TYPE", "beauty")
IS_LAMBDA = os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None

# Import only beauty-related routers (NO ML, NO PyTorch)
from api.endpoints.hair_color import router as hair_color_router
from api.endpoints.usage import router as usage_router

# Lambda initialization flag
_lambda_initialized = False


# ========== Initialize Sentry (if configured) ==========
sentry_enabled = init_sentry()
if sentry_enabled:
    logger.info("Sentry error tracking enabled")


# ========== Rate Limiter Initialization ==========
limiter = Limiter(key_func=get_remote_address)

# ========== Service Startup Status Tracking ==========
# "gemini" = GEMINI_API_KEY 설정됨 (google-genai Client 는 호출 시점에 생성)
startup_status = {"gemini": False}


# ========== FastAPI App Initialization ==========
def _docs_kwargs(environment: str) -> dict:
    """Interactive API docs are disabled in production."""
    if environment == "production":
        return {"docs_url": None, "redoc_url": None, "openapi_url": None}
    return {"docs_url": "/docs", "redoc_url": "/redoc", "openapi_url": "/openapi.json"}


app = FastAPI(
    title="BeautyMe Beauty API",
    description="Hair Color Recommendation + Synthesis (Lightweight)",
    version="1.0.0",
    **_docs_kwargs(settings.ENVIRONMENT),
)

# Attach limiter to app state
app.state.limiter = limiter
# 메인 Lambda(main.py)와 같은 한국어 429 본문 (slowapi 기본 핸들러는 영어 + 제한 수치 노출)
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)


# ========== Trusted Host Middleware ==========
allowed_hosts = ["*"]
app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)


# ========== CORS Middleware ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)


# ========== File Size Limit Middleware ==========
# NOTE: registered BEFORE add_security_headers so the security headers
# middleware wraps it (later registrations run *outside* earlier ones).
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """Limit file upload size to prevent DoS attacks"""
    if request.method == "POST":
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                declared_size = int(content_length)
            except (TypeError, ValueError):
                declared_size = 0

            if declared_size > MAX_FILE_SIZE:
                logger.warning(
                    f"File too large: {declared_size} bytes (max: {MAX_FILE_SIZE})"
                )
                # HTTPException raised inside http middleware is not handled by
                # ExceptionMiddleware, so it would surface as a plain 500.
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
                    },
                )
    return await call_next(request)


# ========== Security Headers Middleware ==========
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)

    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data: https:; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"

    if request.url.scheme == "https" or settings.ENVIRONMENT == "production":
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )

    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "geolocation=(), microphone=(), camera=(), payment=(), usb=()"
    )

    if "Server" in response.headers:
        del response.headers["Server"]

    return response


# ========== Core Service Initialization (shared) ==========
def _init_core_services(strict_db: bool = False) -> None:
    """Gemini 키 확인 / DynamoDB 초기화 (로컬 uvicorn startup 과 Lambda 공용)"""
    if not settings.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set!")
        raise RuntimeError("GEMINI_API_KEY environment variable is required")

    startup_status["gemini"] = True
    logger.info("Gemini API key configured")

    try:
        from database.dynamodb_connection import init_dynamodb

        init_dynamodb()
        logger.info("Database initialized")
    except Exception as e:
        if strict_db:
            raise
        logger.warning(f"Database initialization failed: {str(e)}")


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
# lifespan="off": Mangum 은 invocation 마다 LifespanCycle 을 생성하므로
# lifespan="on" 이면 startup 이벤트가 매 요청마다 재실행된다.
# Lambda 초기화는 lambda_init_middleware → ensure_lambda_initialization() 담당.
MANGUM_LIFESPAN = "off"

try:
    from mangum import Mangum

    handler = Mangum(app, lifespan=MANGUM_LIFESPAN)
    logger.info(f"Beauty Lambda handler initialized (lifespan={MANGUM_LIFESPAN})")
except ImportError:
    logger.warning("Mangum not installed - Lambda handler not available")
    handler = None


# ========== Main Entry Point ==========
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
