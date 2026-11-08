"""
두 Lambda 엔트리(main.py = hairme-analyze, main_beauty.py = hairme-beauty)가
공유하는 FastAPI 앱 구성.

create_app() 이 하는 일 (순서 그대로):
  1. Sentry 초기화
  2. FastAPI 생성 (production 에서는 /docs·/redoc·/openapi.json 비활성)
  3. 레이트리미터를 app.state 에 붙이고 한국어 429 핸들러 등록
  4. 미들웨어 등록: TrustedHost → CORS → 업로드 크기 제한 → 보안 헤더

Starlette 미들웨어는 나중에 등록한 것이 바깥을 감싼다. 그래서 보안 헤더가
업로드 크기 제한(413)을 감싸 413 응답에도 보안 헤더가 붙는다.
엔트리 파일은 create_app() 뒤에 콜드스타트 초기화(lambda_init_middleware)를
등록하므로 그것이 가장 바깥이 된다.

라우터·헬스·루트 응답·콜드스타트 초기화처럼 앱마다 다른 부분은 엔트리 파일에 둔다.
이 모듈은 무거운 ML 라이브러리를 import 하지 않는다 (tests/test_import_footprint.py).
"""

from typing import Callable, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config.settings import settings
from core.logging import logger
from core.monitoring import init_sentry
from core.rate_limit import rate_limit_exceeded_handler

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# TrustedHostMiddleware 는 ALB 헬스 체크 문제로 전체 허용 (커스텀 도메인 설정 후 재검토)
ALLOWED_HOSTS = ["*"]

# lifespan="off": Mangum 은 invocation 마다 LifespanCycle 을 새로 만들기 때문에
# lifespan="on" 이면 startup 이벤트(핵심 서비스 초기화)가 매 요청마다 재실행된다.
# Lambda 초기화는 각 엔트리의 lambda_init_middleware 가 컨테이너당 1회 수행한다.
MANGUM_LIFESPAN = "off"


def docs_kwargs(environment: str) -> dict:
    """
    Interactive API docs are disabled in production.

    Swagger UI / ReDoc / openapi.json expose the full internal API surface
    (admin routes, request schemas), so they are only served outside production.
    """
    if environment == "production":
        return {"docs_url": None, "redoc_url": None, "openapi_url": None}
    return {"docs_url": "/docs", "redoc_url": "/redoc", "openapi_url": "/openapi.json"}


def init_monitoring() -> bool:
    """Sentry 를 초기화하고 결과를 로그로 남긴다."""
    enabled = init_sentry()
    if enabled:
        logger.info("✅ Sentry error tracking enabled")
    else:
        logger.info("ℹ️  Sentry not configured - running without error tracking")
    return enabled


# ========== HTTP 미들웨어 ==========
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
                    f"🚫 File too large: {declared_size} bytes (max: {MAX_FILE_SIZE})"
                )
                # NOTE: HTTPException raised inside an http middleware is not
                # handled by ExceptionMiddleware (it runs further in), so it
                # would surface as a plain 500. Return a response instead.
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
                    },
                )
    return await call_next(request)


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

    # HSTS - only in HTTPS/production environments
    if request.url.scheme == "https" or settings.ENVIRONMENT == "production":
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )

    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "geolocation=(), microphone=(), camera=(), payment=(), usb=()"
    )

    # MutableHeaders 에는 pop 이 없어 del 로 지운다
    if "Server" in response.headers:
        del response.headers["Server"]

    return response


# ========== 앱 생성 ==========
def create_app(*, title: str, description: str, version: str) -> FastAPI:
    """공통 구성(Sentry·문서 노출·429·미들웨어)을 갖춘 FastAPI 앱을 만든다."""
    init_monitoring()

    app = FastAPI(
        title=title,
        description=description,
        version=version,
        **docs_kwargs(settings.ENVIRONMENT),
    )

    # 429 본문은 앱 화면에 그대로 노출되므로 slowapi 기본(영어) 핸들러를 쓰지 않는다.
    app.state.limiter = Limiter(key_func=get_remote_address)
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    # 등록 순서 = 안쪽 → 바깥쪽 (나중에 등록한 것이 바깥)
    if settings.ENVIRONMENT == "production":
        logger.info(f"🔒 Trusted hosts: {ALLOWED_HOSTS}")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    )
    # 보안 헤더가 413 응답도 감싸도록 업로드 제한을 먼저 등록한다.
    app.middleware("http")(limit_upload_size)
    app.middleware("http")(add_security_headers)

    return app


# ========== 핵심 서비스 초기화 ==========
def init_core_services(startup_status: dict, strict_db: bool = False) -> None:
    """
    Gemini 키 확인 / DynamoDB 초기화 (로컬 uvicorn startup 과 Lambda 공용).

    Args:
        startup_status: 엔트리 모듈의 상태 dict. 키 확인 시 "gemini" 를 True 로 바꾼다.
        strict_db: True 면 DB 초기화 실패를 그대로 전파한다 (로컬 개발용).
                   False 면 WARNING 만 남기고 계속 진행한다 (Lambda 가용성 우선).
    """
    if not settings.GEMINI_API_KEY:
        logger.error("❌ GEMINI_API_KEY is not set!")
        raise RuntimeError("GEMINI_API_KEY environment variable is required")

    startup_status["gemini"] = True
    logger.info("✅ Gemini API 키 확인 완료")

    try:
        from database.dynamodb_connection import init_dynamodb

        logger.info("🔄 Initializing DynamoDB connection...")
        init_dynamodb()
        logger.info("✅ Database initialized")
    except Exception as e:
        if strict_db:
            raise
        logger.warning(f"⚠️ Database initialization failed: {str(e)}")


# ========== Lambda 핸들러 ==========
def create_mangum_handler(app: FastAPI, name: str) -> Optional[Callable]:
    """Mangum 어댑터를 만든다. mangum 이 없으면(로컬 venv) None."""
    try:
        from mangum import Mangum
    except ImportError:
        logger.warning(f"⚠️ Mangum not installed - {name} handler not available")
        return None

    mangum_handler = Mangum(app, lifespan=MANGUM_LIFESPAN)
    logger.info(f"✅ {name} handler initialized (lifespan={MANGUM_LIFESPAN})")
    return mangum_handler
