"""
HairMe Backend - AI-powered Hairstyle Recommendation Service
Version: 20.2.0 (MediaPipe transition complete)
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config.settings import settings
from core.logging import logger, log_structured
from core.cache import init_redis
from core.ml_loader import load_ml_model, load_sentence_transformer
from core.dependencies import init_services
from core.monitoring import init_sentry
from database.connection import init_database
from database.migration import migrate_database_schema
from models.mediapipe_analyzer import MediaPipeFaceAnalyzer
from services.hybrid_recommender import create_hybrid_service
from services.feedback_collector import get_feedback_collector
from services.retrain_queue import get_retrain_queue
from routers.admin import router as admin_router
from api.endpoints.analyze import router as analyze_router
# from api.endpoints.analyze_improved import router as analyze_improved_router  # Disabled: requires hybrid_recommender_improved
from api.endpoints.feedback import router as feedback_router
import google.generativeai as genai


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
    """Initialize all services on server startup"""
    logger.info("🚀 서버 시작 중...")

    # ========== 1. Gemini API 키 검증 (필수) ==========
    if not settings.GEMINI_API_KEY:
        logger.error("❌ GEMINI_API_KEY is not set!")
        raise RuntimeError("GEMINI_API_KEY environment variable is required")

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        startup_status["gemini"] = True
        logger.info("✅ Gemini API 초기화 완료")
    except Exception as e:
        logger.error(f"❌ Gemini API 초기화 실패: {str(e)}")
        raise RuntimeError(f"Gemini API initialization failed: {str(e)}")

    # ========== 2. MediaPipe Face Analyzer (필수) ==========
    try:
        mediapipe_analyzer = MediaPipeFaceAnalyzer()
        startup_status["mediapipe"] = True
        logger.info("✅ MediaPipe 얼굴 분석기 초기화 완료")
        log_structured("mediapipe_initialized", {"status": "success", "landmarks": 478})
    except Exception as e:
        logger.error(f"❌ MediaPipe 초기화 실패: {str(e)}")
        raise RuntimeError(f"MediaPipe initialization failed: {str(e)}")

    # ========== 3. ML Model (선택 - 실패해도 진행) ==========
    ml_loaded = load_ml_model()
    startup_status["ml_model"] = ml_loaded
    if ml_loaded:
        logger.info("✅ ML 모드: 활성화")
        log_structured("ml_model_loaded", {"status": "success", "model_path": settings.ML_MODEL_PATH})
    else:
        logger.warning("⚠️ ML 모드: 비활성화 (기본 점수 사용)")
        log_structured("ml_model_loaded", {"status": "failed", "fallback": "default_score"})

    # ========== 4. Sentence Transformer (선택 - 실패해도 진행) ==========
    st_loaded = load_sentence_transformer()
    startup_status["sentence_transformer"] = st_loaded
    if st_loaded:
        logger.info("✅ 스타일 임베딩: 활성화")
        log_structured("sentence_transformer_loaded", {
            "status": "success",
            "model": settings.SENTENCE_TRANSFORMER_MODEL,
            "embedding_dim": 384
        })
    else:
        logger.warning("⚠️ 스타일 임베딩: 비활성화")

    # ========== 5. Hybrid Service (필수 - Gemini가 있으므로) ==========
    try:
        hybrid_service = create_hybrid_service(settings.GEMINI_API_KEY)
        startup_status["hybrid_service"] = True
        logger.info("✅ 하이브리드 추천 서비스 초기화 완료")
    except Exception as e:
        logger.error(f"❌ 하이브리드 서비스 초기화 실패: {str(e)}")
        raise RuntimeError(f"Hybrid service initialization failed: {str(e)}")

    # ========== 6. Feedback Collector (선택) ==========
    feedback_collector = None
    try:
        feedback_collector = get_feedback_collector()
        startup_status["feedback_collector"] = True
        logger.info("✅ 피드백 수집기 초기화 완료")
    except Exception as e:
        logger.error(f"❌ 피드백 수집기 초기화 실패: {str(e)}")
        # 선택사항이므로 계속 진행

    # ========== 7. Retrain Queue (선택) ==========
    retrain_queue = None
    try:
        retrain_queue = get_retrain_queue()
        startup_status["retrain_queue"] = True
        logger.info("✅ 재학습 큐 초기화 완료")
    except Exception as e:
        logger.error(f"❌ 재학습 큐 초기화 실패: {str(e)}")
        # 선택사항이므로 계속 진행

    # ========== 8. 의존성 주입 초기화 ==========
    init_services(
        mediapipe_analyzer=mediapipe_analyzer,
        hybrid_service=hybrid_service,
        feedback_collector=feedback_collector if startup_status["feedback_collector"] else None,
        retrain_queue=retrain_queue if startup_status["retrain_queue"] else None
    )

    # ========== 9. Database & Cache ==========
    db_initialized = init_database()
    if db_initialized:
        migrate_database_schema()

    init_redis()

    # ========== 10. 초기화 상태 로깅 ==========
    logger.info(f"📊 서비스 초기화 상태: {startup_status}")


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
    handler = Mangum(app, lifespan="off")
    logger.info("✅ Lambda handler initialized")
except ImportError:
    logger.warning("⚠️ Mangum not installed - Lambda handler not available")
    handler = None


# ========== Main Entry Point ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
