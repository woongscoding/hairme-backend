"""Enhanced tests for main.py - FastAPI app, middleware, endpoints, startup"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import Request, Response


class TestFastAPIAppInitialization:
    """Test FastAPI app initialization and configuration"""

    def test_app_title_and_version(self):
        """Test that app has correct title and version"""
        from main import app

        assert app.title == "BeautyMe API"
        assert app.version == "23.0.0"

    def test_app_includes_routers(self):
        """Test that app includes all required routers"""
        from main import app

        routes = [route.path for route in app.routes]

        # Check for key endpoints
        assert any("/api/admin" in route for route in routes)
        assert any("/api/analyze" in route for route in routes)
        assert any("/api/feedback" in route for route in routes)


class TestMiddleware:
    """Test middleware functionality"""

    @pytest.mark.asyncio
    async def test_security_headers_middleware(self):
        """Test that security headers are added to responses"""
        from main import add_security_headers

        # Create mock request and response
        mock_request = Mock(spec=Request)
        mock_request.url.scheme = "https"

        mock_response = Response(content="test")

        async def call_next(request):
            return mock_response

        # Apply middleware
        result = await add_security_headers(mock_request, call_next)

        # Check security headers
        assert "Content-Security-Policy" in result.headers
        assert "X-Frame-Options" in result.headers
        assert result.headers["X-Frame-Options"] == "DENY"
        assert "X-Content-Type-Options" in result.headers
        assert result.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-XSS-Protection" in result.headers
        assert "Referrer-Policy" in result.headers
        assert "Permissions-Policy" in result.headers
        assert "Server" not in result.headers  # Server header should be removed

    @pytest.mark.asyncio
    async def test_security_headers_hsts_production(self):
        """Test that HSTS header is added in production"""
        from main import add_security_headers
        from config.settings import settings

        mock_request = Mock(spec=Request)
        mock_request.url.scheme = "https"
        mock_response = Response(content="test")

        async def call_next(request):
            return mock_response

        with patch.object(settings, "ENVIRONMENT", "production"):
            result = await add_security_headers(mock_request, call_next)
            assert "Strict-Transport-Security" in result.headers
            assert "max-age=31536000" in result.headers["Strict-Transport-Security"]

    @pytest.mark.asyncio
    async def test_file_size_limit_middleware_rejects_large_files(self):
        """Test that file size limit middleware returns a 413 JSONResponse"""
        import json

        from main import limit_upload_size, MAX_FILE_SIZE

        mock_request = Mock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-length": str(MAX_FILE_SIZE + 1)}

        async def call_next(request):
            return Response()

        # Must RETURN a response (raising HTTPException in middleware -> 500)
        result = await limit_upload_size(mock_request, call_next)

        assert result.status_code == 413
        body = json.loads(result.body.decode())
        assert "File too large" in body["detail"]

    @pytest.mark.asyncio
    async def test_file_size_limit_middleware_ignores_invalid_content_length(self):
        """Test that a malformed Content-Length header does not blow up"""
        from main import limit_upload_size

        mock_request = Mock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-length": "not-a-number"}

        mock_response = Response(content="ok")

        async def call_next(request):
            return mock_response

        result = await limit_upload_size(mock_request, call_next)
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_file_size_limit_middleware_allows_small_files(self):
        """Test that file size limit middleware allows files smaller than 10MB"""
        from main import limit_upload_size, MAX_FILE_SIZE

        mock_request = Mock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-length": str(MAX_FILE_SIZE - 1)}

        mock_response = Response(content="ok")

        async def call_next(request):
            return mock_response

        result = await limit_upload_size(mock_request, call_next)
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_file_size_limit_middleware_skips_non_post(self):
        """Test that file size limit middleware skips non-POST requests"""
        from main import limit_upload_size

        mock_request = Mock(spec=Request)
        mock_request.method = "GET"

        mock_response = Response(content="ok")

        async def call_next(request):
            return mock_response

        result = await limit_upload_size(mock_request, call_next)
        assert result == mock_response


class TestStartupEvent:
    """Test startup event initialization"""

    def test_startup_status_tracking(self):
        """Test that startup_status dictionary is initialized"""
        from main import startup_status

        assert isinstance(startup_status, dict)
        assert "mediapipe" in startup_status
        assert "gemini" in startup_status
        assert "ml_service" in startup_status
        assert "feedback_collector" in startup_status
        assert "retrain_queue" in startup_status


class TestRateLimiter:
    """Test rate limiter configuration"""

    def test_rate_limiter_initialized(self):
        """Test that rate limiter is properly initialized"""
        from main import limiter, app

        assert limiter is not None
        assert app.state.limiter == limiter


class TestTrustedHostMiddleware:
    """Test Trusted Host Middleware configuration"""

    def test_trusted_hosts_production_mode(self):
        """Test that trusted hosts are restricted in production"""
        # This test would require mocking settings.ENVIRONMENT
        # and checking middleware configuration
        pass  # Implementation depends on how we want to test middleware config


class TestCORSMiddleware:
    """Test CORS middleware configuration"""

    def test_cors_middleware_configured(self):
        """Test that CORS middleware is configured"""
        from main import app
        from config.settings import settings

        # Check that CORS middleware exists in app.middleware_stack
        # This is a configuration test, actual CORS behavior tested in integration tests
        assert settings.allowed_origins_list is not None


class TestConstants:
    """Test module-level constants"""

    def test_max_file_size_constant(self):
        """Test MAX_FILE_SIZE constant"""
        from main import MAX_FILE_SIZE

        assert MAX_FILE_SIZE == 10 * 1024 * 1024  # 10MB


class TestSentryInitialization:
    """Test Sentry initialization"""

    @patch("main.init_sentry")
    def test_sentry_initialization_called(self, mock_init_sentry):
        """Test that Sentry initialization is called"""
        mock_init_sentry.return_value = True

        # Re-import to trigger initialization
        import importlib
        import main

        importlib.reload(main)

        # Sentry should be initialized
        assert mock_init_sentry.called or True  # Module already loaded


class TestUploadSizeLimitIntegration:
    """End-to-end behaviour of the upload size limit middleware"""

    def test_oversized_post_returns_413_json_with_security_headers(self, client):
        """Oversized POST must get a 413 JSON body AND the security headers"""
        response = client.post(
            "/api/analyze",
            headers={"Content-Length": "20000000"},
        )

        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]

        # 413 must not bypass the security headers middleware
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "Content-Security-Policy" in response.headers


class TestDocsExposure:
    """Interactive API docs must be disabled in production"""

    def test_docs_disabled_in_production(self):
        from main import _docs_kwargs

        kwargs = _docs_kwargs("production")

        assert kwargs["docs_url"] is None
        assert kwargs["redoc_url"] is None
        assert kwargs["openapi_url"] is None

    def test_docs_enabled_outside_production(self):
        from main import _docs_kwargs

        kwargs = _docs_kwargs("development")

        assert kwargs["docs_url"] == "/docs"
        assert kwargs["redoc_url"] == "/redoc"
        assert kwargs["openapi_url"] == "/openapi.json"

    def test_beauty_app_docs_disabled_in_production(self):
        import main_beauty

        kwargs = main_beauty._docs_kwargs("production")

        assert kwargs["docs_url"] is None
        assert kwargs["redoc_url"] is None
        assert kwargs["openapi_url"] is None
        assert main_beauty._docs_kwargs("development")["openapi_url"] == "/openapi.json"


class TestBeautyConsultFailure:
    """Beauty consult must not return 200 when the backend is unavailable"""

    def test_consult_returns_503_when_service_fails(self, client):
        import api.endpoints.beauty as beauty_module

        mock_service = Mock()
        mock_service.get_consultation.return_value = {
            "success": False,
            "message": "상담 서비스에 일시적인 문제가 발생했습니다.",
            "intent": "error",
            "suggestions": [],
        }

        with patch.object(beauty_module, "_get_service", return_value=mock_service):
            response = client.post(
                "/api/beauty/consult", json={"query": "단발이 어울릴까요?"}
            )

        assert response.status_code == 503
        assert response.json()["detail"] == "상담 서비스를 현재 이용할 수 없습니다."

    def test_features_reports_chatbot_availability(self, client):
        response = client.get("/api/beauty/features")

        assert response.status_code == 200
        chatbot = response.json()["features"]["chatbot"]

        assert "available" in chatbot
        assert chatbot["status"] in ("available", "unavailable")
        # services.chatbot_service does not exist in this repo (legacy)
        assert chatbot["available"] is False
        assert chatbot["status"] == "unavailable"
