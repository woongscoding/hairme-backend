"""
HairMe Backend - AI-powered Hairstyle Recommendation Service
Version: 20.2.0 (MediaPipe transition complete)
"""

import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config.settings import settings
from core.logging import logger, log_structured
from core.monitoring import init_sentry

# Lambda 환경 감지 - 무거운 import를 지연 로딩
IS_LAMBDA = os.environ.get('AWS_LAMBDA_FUNCTION_NAME') is not None

# 경량 import만 즉시 로드
from routers.admin import router as admin_router
from api.endpoints.analyze import router as analyze_router
from api.endpoints.feedback import router as feedback_router

# 무거운 모듈은 필요할 때 로드 (Lambda init 타임아웃 방지)
genai = None
MediaPipeFaceAnalyzer = None
create_hybrid_service = None
get_feedback_collector = None
get_retrain_queue = None

# Lambda initialization flag
_lambda_initialized = False


# ========== Initialize Sentry (if configured) ==========
sentry_enabled = init_sentry()
if sentry_enabled:
    logger.info("✅ Sentry error tracking enabled")
else:
    logger.info("ℹ️  Sentry not configured - running without error tracking")


# ========== Rate Limiter Initialization ==========
limiter = Limiter(key_func=get_remote_address)

# ========== Service Startup Status Tracking ==========
startup_status = {
    "mediapipe": False,
    "gemini": False,
    "ml_model": False,
    "sentence_transformer": False,
    "hybrid_service": False,
    "feedback_collector": False,
    "retrain_queue": False
}

# ========== FastAPI App Initialization ==========
app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION
)

# Attach limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ========== Trusted Host Middleware ==========
# Prevent Host Header Injection attacks
allowed_hosts = ["*"]  # Default: allow all
if settings.ENVIRONMENT == "production":
    # Production: Allow all hosts (TrustedHostMiddleware causes issues with ALB health checks)
    # TODO: Revisit this once we configure custom domain
    allowed_hosts = ["*"]
    logger.info(f"🔒 Trusted hosts: {allowed_hosts}")

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=allowed_hosts
)


# ========== CORS Middleware ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== Security Headers Middleware ==========
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """
    Add security headers to all responses

    Headers added:
    - Content-Security-Policy: Prevent XSS attacks
    - X-Frame-Options: Prevent clickjacking
    - X-Content-Type-Options: Prevent MIME sniffing
    - X-XSS-Protection: Enable browser XSS protection (legacy)
    - Strict-Transport-Security: Force HTTPS (production only)
    - Referrer-Policy: Control referrer information
    - Permissions-Policy: Restrict browser features
    """
    response = await call_next(request)

    # Content Security Policy
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data: https:; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )

    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"

    # Prevent MIME type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"

    # XSS Protection (legacy browsers)
    response.headers["X-XSS-Protection"] = "1; mode=block"

    # HSTS - only in HTTPS/production environments
    if request.url.scheme == "https" or settings.ENVIRONMENT == "production":
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )

    # Referrer Policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Permissions Policy (restrict browser features)
    response.headers["Permissions-Policy"] = (
        "geolocation=(), microphone=(), camera=(), payment=(), usb=()"
    )

    # Remove server header for security (MutableHeaders doesn't have pop method)
    # response.headers.pop("Server", None)  # Commented out - not supported
    # Alternative: Override the Server header instead
    if "Server" in response.headers:
        del response.headers["Server"]

    return response


# ========== Lambda Initialization Helper ==========
def ensure_lambda_initialization():
    """Ensure essential services are initialized for Lambda"""
    global _lambda_initialized

    if _lambda_initialized:
        return

    if not IS_LAMBDA:
        return

    logger.info("🔧 Lambda cold start - initializing essential services...")

    # Initialize Gemini API (essential)
    if not settings.GEMINI_API_KEY:
        logger.error("❌ GEMINI_API_KEY is not set!")
        raise RuntimeError("GEMINI_API_KEY environment variable is required")

    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        startup_status["gemini"] = True
        logger.info("✅ Gemini API configured for Lambda")
    except Exception as e:
        logger.error(f"❌ Gemini API setup failed: {str(e)}")
        raise RuntimeError(f"Gemini API initialization failed: {str(e)}")

    # Initialize Database & Cache
    try:
        from database import init_database
        from core.cache import init_redis

        db_initialized = init_database()
        if db_initialized:
            use_dynamodb = os.environ.get('USE_DYNAMODB', 'false').lower() == 'true'
            if not use_dynamodb:
                from database.migration import migrate_database_schema
                migrate_database_schema()

        init_redis()
        logger.info("✅ Database and cache initialized for Lambda")
    except Exception as e:
        logger.warning(f"⚠️ Database/cache initialization failed: {str(e)}")

    _lambda_initialized = True
    logger.info("✅ Lambda initialization complete")


# ========== Lambda Initialization Middleware ==========
@app.middleware("http")
async def lambda_init_middleware(request: Request, call_next):
    """Ensure Lambda is initialized before processing requests"""
    if IS_LAMBDA and not _lambda_initialized:
        ensure_lambda_initialization()
    return await call_next(request)


# ========== File Size Limit Middleware ==========
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """Limit file upload size to prevent DoS attacks"""
    if request.method == "POST":
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_FILE_SIZE:
            logger.warning(f"🚫 File too large: {int(content_length)} bytes (max: {MAX_FILE_SIZE})")
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            )
    return await call_next(request)


# ========== Register Routers ==========
app.include_router(admin_router, prefix="/api", tags=["admin"])
app.include_router(analyze_router, prefix="/api", tags=["analysis"])
# app.include_router(analyze_improved_router, prefix="/api", tags=["analysis_improved"])  # Disabled: requires hybrid_recommender_improved
app.include_router(feedback_router, prefix="/api", tags=["feedback"])


# ========== Startup Event ==========
@app.on_event("startup")
async def startup_event():
    """Initialize essential services on server startup"""
    logger.info("🚀 서버 시작 중 (Lazy Loading 적용됨)...")

    # ========== 1. Gemini API 키 검증 (필수) ==========
    if not settings.GEMINI_API_KEY:
        logger.error("❌ GEMINI_API_KEY is not set!")
        raise RuntimeError("GEMINI_API_KEY environment variable is required")

    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        startup_status["gemini"] = True
        logger.info("✅ Gemini API 설정 완료")
    except Exception as e:
        logger.error(f"❌ Gemini API 설정 실패: {str(e)}")
        raise RuntimeError(f"Gemini API initialization failed: {str(e)}")

    # ========== 2. Database & Cache ==========
    from database import init_database  # Uses DynamoDB or MySQL based on USE_DYNAMODB
    from core.cache import init_redis

    db_initialized = init_database()
    if db_initialized:
        use_dynamodb = os.environ.get('USE_DYNAMODB', 'false').lower() == 'true'
        if not use_dynamodb:
            from database.migration import migrate_database_schema
            migrate_database_schema()

    init_redis()

    logger.info("✅ 기본 서비스 초기화 완료 (AI 모델은 첫 요청 시 로드됩니다)")


# ========== Root Endpoint ==========
@app.get("/")
async def root():
    """Root endpoint with service status"""
    from core.dependencies import _mediapipe_analyzer

    return {
        "message": f"{settings.APP_TITLE} - v{settings.APP_VERSION} (MediaPipe 전환 완료)",
        "version": settings.APP_VERSION,
        "model": settings.MODEL_NAME,
        "status": "running",
        "features": {
            "mediapipe_analysis": "enabled" if _mediapipe_analyzer else "disabled",
            "gemini_analysis": "enabled" if settings.GEMINI_API_KEY else "disabled",
            "redis_cache": "enabled",
            "database": "enabled",
            "feedback_system": "enabled",
            "ml_prediction": "enabled",
            "style_embedding": "enabled"
        }
    }


# ========== Health Check Endpoint ==========
@app.get("/api/health")
async def health_check(deep: bool = False):
    """
    Enhanced health check endpoint with actual service validation

    Query parameters:
    - deep: If true, runs comprehensive checks including Gemini API ping (slower)

    Returns:
    - status: "healthy", "degraded", or "unhealthy"
    - startup: Services initialized during startup
    - checks: Real-time connectivity checks
    - system: CPU, memory, disk metrics
    - circuit_breaker: Circuit breaker state
    """
    from core.health_check import get_health_check_service

    # Basic startup status
    required_services_ok = all([
        startup_status["mediapipe"],
        startup_status["gemini"],
        startup_status["hybrid_service"]
    ])

    base_status = {
        "status": "healthy" if required_services_ok else "degraded",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "startup": {
            "required_services": {
                "mediapipe": startup_status["mediapipe"],
                "gemini": startup_status["gemini"],
                "hybrid_service": startup_status["hybrid_service"]
            },
            "optional_services": {
                "ml_model": startup_status["ml_model"],
                "sentence_transformer": startup_status["sentence_transformer"],
                "feedback_collector": startup_status["feedback_collector"],
                "retrain_queue": startup_status["retrain_queue"]
            }
        }
    }

    # Run comprehensive health checks
    health_service = get_health_check_service()
    comprehensive_result = await health_service.comprehensive_health_check(
        include_expensive_checks=deep
    )

    # Merge results
    base_status.update({
        "checks": comprehensive_result["checks"],
        "check_duration_ms": comprehensive_result["check_duration_ms"],
        "timestamp": comprehensive_result["timestamp"]
    })

    # Update overall status based on checks
    if comprehensive_result["status"] == "degraded":
        base_status["status"] = "degraded"

    return base_status


# ========== Lambda Handler ==========
# For AWS Lambda deployment using Mangum
try:
    from mangum import Mangum
    # lifespan="on" 으로 설정하면 첫 요청 시 startup 이벤트가 실행됨
    # Lambda init 단계에서는 import만 하고, 무거운 작업은 첫 요청 시 수행
    handler = Mangum(app, lifespan="on")
    logger.info("✅ Lambda handler initialized (lifespan=on)")
except ImportError:
    logger.warning("⚠️ Mangum not installed - Lambda handler not available")
    handler = None


# ========== Main Entry Point ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
