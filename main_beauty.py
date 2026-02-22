"""
BeautyMe Backend - Beauty Lambda (Lightweight)
Version: 1.0.0

Features:
- Personal Color Analysis (MediaPipe + ITA algorithm)
- Hair Color Recommendation (Gemini)
- Beauty Consultation (Gemini)

NO PyTorch - Fast cold start (~10s)
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

# Lambda type identifier
LAMBDA_TYPE = os.environ.get('LAMBDA_TYPE', 'beauty')
IS_LAMBDA = os.environ.get('AWS_LAMBDA_FUNCTION_NAME') is not None

# Import only beauty-related routers (NO ML, NO PyTorch)
from api.endpoints.personal_color import router as personal_color_router
from api.endpoints.hair_color import router as hair_color_router
from api.endpoints.beauty import router as beauty_router

# Lambda initialization flag
_lambda_initialized = False


# ========== Initialize Sentry (if configured) ==========
sentry_enabled = init_sentry()
if sentry_enabled:
    logger.info("Sentry error tracking enabled")


# ========== Rate Limiter Initialization ==========
limiter = Limiter(key_func=get_remote_address)

# ========== Service Startup Status Tracking ==========
startup_status = {
    "mediapipe": False,
    "gemini": False
}

# ========== FastAPI App Initialization ==========
app = FastAPI(
    title="BeautyMe Beauty API",
    description="Personal Color + Hair Color Recommendation (Lightweight)",
    version="1.0.0"
)

# Attach limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ========== Trusted Host Middleware ==========
allowed_hosts = ["*"]
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


# ========== Lambda Initialization Helper ==========
def ensure_lambda_initialization():
    """Ensure essential services are initialized for Lambda"""
    global _lambda_initialized

    if _lambda_initialized:
        return

    if not IS_LAMBDA:
        return

    logger.info("Lambda cold start - initializing beauty services...")

    # Initialize Gemini API (essential)
    if not settings.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set!")
        raise RuntimeError("GEMINI_API_KEY environment variable is required")

    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        startup_status["gemini"] = True
        logger.info("Gemini API configured")
    except Exception as e:
        logger.error(f"Gemini API setup failed: {str(e)}")
        raise RuntimeError(f"Gemini API initialization failed: {str(e)}")

    # Initialize Database & Cache (optional)
    try:
        from database import init_database
        from core.cache import init_redis

        init_database()
        init_redis()
        logger.info("Database and cache initialized")
    except Exception as e:
        logger.warning(f"Database/cache initialization failed: {str(e)}")

    _lambda_initialized = True
    logger.info("Beauty Lambda initialization complete")


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
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            )
    return await call_next(request)


# ========== Register Routers (Beauty Only) ==========
app.include_router(personal_color_router, prefix="/api", tags=["personal_color"])
app.include_router(hair_color_router, prefix="/api", tags=["hair_color"])
app.include_router(beauty_router, prefix="/api", tags=["beauty"])


# ========== Startup Event ==========
@app.on_event("startup")
async def startup_event():
    """Initialize essential services on server startup"""
    logger.info("Beauty Lambda starting...")

    # Gemini API key validation
    if not settings.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set!")
        raise RuntimeError("GEMINI_API_KEY environment variable is required")

    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        startup_status["gemini"] = True
        logger.info("Gemini API configured")
    except Exception as e:
        logger.error(f"Gemini API setup failed: {str(e)}")
        raise RuntimeError(f"Gemini API initialization failed: {str(e)}")

    # Database & Cache
    from database import init_database
    from core.cache import init_redis

    init_database()
    init_redis()

    logger.info("Beauty Lambda startup complete")


# ========== Root Endpoint ==========
@app.get("/")
async def root():
    """Root endpoint with service status"""
    return {
        "name": "BeautyMe Beauty API",
        "message": "Personal Color + Hair Color Recommendation",
        "version": "1.0.0",
        "lambda_type": LAMBDA_TYPE,
        "status": "running",
        "features": {
            "personal_color": "enabled (ITA algorithm)",
            "hair_color_recommendation": "enabled",
            "beauty_consultation": "enabled",
            "hair_color_synthesis": "enabled" if settings.GEMINI_API_KEY else "disabled"
        },
        "endpoints": {
            "personal_color": "/api/personal-color",
            "hair_color": "/api/hair-color/{type}",
            "beauty_analyze": "/api/beauty/analyze",
            "beauty_consult": "/api/beauty/consult"
        }
    }


# ========== Health Check Endpoint ==========
@app.get("/api/health")
async def health_check():
    """Simple health check for Beauty Lambda"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "lambda_type": LAMBDA_TYPE,
        "environment": settings.ENVIRONMENT,
        "services": {
            "gemini": startup_status["gemini"],
            "mediapipe": startup_status["mediapipe"]
        }
    }


# ========== Lambda Handler ==========
try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="on")
    logger.info("Beauty Lambda handler initialized")
except ImportError:
    logger.warning("Mangum not installed - Lambda handler not available")
    handler = None


# ========== Main Entry Point ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
