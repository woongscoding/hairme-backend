"""core.app_factory: 두 Lambda 앱(main / main_beauty)이 같은 공통 구성을 갖는지 검증

- 미들웨어 순서 (나중에 등록한 것이 바깥): lambda_init → 보안 헤더 → 업로드 제한 → CORS → TrustedHost
- 413 / 429 응답에도 보안 헤더가 붙고, 429 본문은 한국어
- /api/health 는 status "healthy" + 200 (배포 게이트)
"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.testclient import TestClient
from slowapi.errors import RateLimitExceeded

import main
import main_beauty
from core import app_factory
from core.rate_limit import rate_limit_exceeded_handler, rate_limit_response_body

APPS = {"main": main, "main_beauty": main_beauty}


def _assert_security_headers(response):
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "Content-Security-Policy" in response.headers
    assert "Referrer-Policy" in response.headers
    assert "Permissions-Policy" in response.headers


@pytest.fixture(params=sorted(APPS))
def entry(request):
    return APPS[request.param]


def test_middleware_stack_order(entry):
    stack = entry.app.user_middleware  # 바깥 → 안쪽

    dispatches = [m.kwargs["dispatch"] for m in stack[:3]]
    assert dispatches == [
        entry.lambda_init_middleware,
        app_factory.add_security_headers,
        app_factory.limit_upload_size,
    ]
    assert [m.cls for m in stack[3:]] == [CORSMiddleware, TrustedHostMiddleware]
    assert stack[4].kwargs["allowed_hosts"] == ["*"]


def test_korean_429_handler_registered(entry):
    assert entry.app.exception_handlers[RateLimitExceeded] is (
        rate_limit_exceeded_handler
    )


def test_oversized_post_returns_413_with_security_headers(entry):
    client = TestClient(entry.app)
    response = client.post(
        "/api/hair-color/synthesize", headers={"Content-Length": "20000000"}
    )

    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]
    _assert_security_headers(response)


def test_health_is_healthy(entry):
    healthy = AsyncMock(
        return_value={
            "status": "healthy",
            "timestamp": "2026-11-08T00:00:00",
            "checks": {},
            "check_duration_ms": 1,
        }
    )
    with patch("core.health_check.get_health_check_service") as svc:
        svc.return_value.comprehensive_health_check = healthy
        entry.app.state.limiter.reset()
        with TestClient(entry.app) as client:  # startup 이벤트로 gemini 확인
            response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    _assert_security_headers(response)


def test_rate_limited_route_returns_korean_429_with_security_headers():
    """factory 로 만든 앱에서 실제 제한을 넘기면 한국어 429 + 보안 헤더"""
    app = app_factory.create_app(title="t", description="d", version="0")
    limiter = app.state.limiter

    @app.get("/limited")
    @limiter.limit("1/minute")
    async def limited(request: Request):
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/limited").status_code == 200

    response = client.get("/limited")
    assert response.status_code == 429
    assert response.json() == rate_limit_response_body()
    assert "Rate limit exceeded" not in response.text
    _assert_security_headers(response)


def test_docs_disabled_in_production():
    with patch.object(app_factory.settings, "ENVIRONMENT", "production"):
        app = app_factory.create_app(title="t", description="d", version="0")

    assert app.docs_url is None and app.redoc_url is None and app.openapi_url is None
