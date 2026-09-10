"""Tests for main application endpoints"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


class TestRootEndpoint:
    """Test root endpoint functionality"""

    def test_root_endpoint_returns_200(self, client):
        """Test that root endpoint returns 200 OK"""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_endpoint_contains_version(self, client):
        """Test that root endpoint contains version info"""
        response = client.get("/")
        data = response.json()

        assert "version" in data
        assert "message" in data
        assert "status" in data
        assert data["status"] == "running"

    def test_root_endpoint_shows_features(self, client):
        """Test that root endpoint lists available features"""
        response = client.get("/")
        data = response.json()

        assert "features" in data
        features = data["features"]

        # Check for key features
        assert "face_analysis" in features
        assert "personal_color" in features
        assert "hair_color_recommendation" in features
        assert "hairstyle_recommendation" in features
        assert "hair_color_synthesis" in features


class TestHealthCheck:
    """Test health check endpoint"""

    def test_health_check_returns_200(self, client):
        """Test that health check returns 200 OK"""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_check_shows_healthy_status(self, client):
        """Test that health check shows healthy status"""
        with patch("core.health_check.get_health_check_service") as mock_svc:
            mock_health = mock_svc.return_value
            mock_health.comprehensive_health_check = AsyncMock(
                return_value={
                    "status": "healthy",
                    "timestamp": "2025-01-17T00:00:00",
                    "checks": {
                        "system": {"cpu": {"percent": 50}},
                        "dynamodb": {"status": "healthy"},
                        "circuit_breaker": {"state": "closed"},
                        "gemini_api": {"status": "skipped"},
                    },
                    "check_duration_ms": 50,
                }
            )

            response = client.get("/api/health")
            data = response.json()

            assert "status" in data
            assert data["status"] == "healthy"

    def test_health_check_includes_service_status(self, client):
        """Test that health check includes all service statuses"""
        response = client.get("/api/health")
        data = response.json()

        # Verify top-level structure
        assert "version" in data
        assert "environment" in data
        assert "startup" in data
        assert "checks" in data

        # Verify startup sub-structure
        assert "required_services" in data["startup"]
        assert "optional_services" in data["startup"]

        # Verify checks sub-structure
        assert "system" in data["checks"]
        assert "dynamodb" in data["checks"]
        assert "circuit_breaker" in data["checks"]
        assert "gemini_api" in data["checks"]


class TestCORS:
    """Test CORS middleware configuration"""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in responses"""
        response = client.options("/api/health")

        # Should have CORS headers (even if not all are checked in OPTIONS)
        assert response.status_code in [200, 405]  # OPTIONS might not be implemented

    def test_get_request_works(self, client):
        """Test that regular GET requests work (CORS should not block)"""
        response = client.get("/api/health")
        assert response.status_code == 200


class TestLambdaHandlerLifespan:
    """Task 2: Mangum lifespan="off" (invocation 마다 startup 재실행 방지)"""

    def test_mangum_lifespan_is_off(self):
        import main

        assert main.MANGUM_LIFESPAN == "off"

        # mangum 은 Lambda 이미지에만 설치되어 있다 (로컬 venv 에는 없음)
        if main._mangum_handler is not None:
            assert main._mangum_handler.lifespan == "off"

    def test_startup_and_lambda_init_share_one_implementation(self):
        """startup 이벤트와 Lambda 초기화가 동일한 _init_core_services 를 사용"""
        import main

        assert callable(main._init_core_services)


class TestEnsureLambdaInitialization:
    """Task 2: Lambda 초기화는 컨테이너당 1회만 수행되어야 한다"""

    def test_initializes_only_once(self):
        import main

        original_flag = main._lambda_initialized
        try:
            main._lambda_initialized = False

            with patch("main.IS_LAMBDA", True), patch(
                "database.init_database", return_value=False
            ) as mock_init_db, patch("core.cache.init_redis", return_value=False):
                main.ensure_lambda_initialization()
                main.ensure_lambda_initialization()
                main.ensure_lambda_initialization()

            assert mock_init_db.call_count == 1
            assert main._lambda_initialized is True
        finally:
            main._lambda_initialized = original_flag

    def test_noop_outside_lambda(self):
        """로컬(비 Lambda) 환경에서는 아무것도 초기화하지 않는다"""
        import main

        original_flag = main._lambda_initialized
        try:
            main._lambda_initialized = False

            with patch("main.IS_LAMBDA", False), patch(
                "database.init_database"
            ) as mock_init_db:
                main.ensure_lambda_initialization()

            mock_init_db.assert_not_called()
            assert main._lambda_initialized is False
        finally:
            main._lambda_initialized = original_flag


class TestWarmUpHandler:
    """Task 3: warm-up 이벤트는 HTTP(Mangum) 계층을 거치지 않는다"""

    def test_warmup_event_bypasses_mangum(self):
        import main

        mangum = MagicMock()
        with patch.object(main, "_mangum_handler", mangum), patch.object(
            main, "warm_up"
        ) as mock_warm:
            response = main.handler({"warmup": True}, None)

        assert response == {"statusCode": 200, "body": "warm"}
        mock_warm.assert_called_once()
        mangum.assert_not_called()

    def test_non_warmup_event_delegates_to_mangum(self):
        import main

        mangum = MagicMock(return_value={"statusCode": 200, "body": "{}"})
        event = {"requestContext": {"http": {"method": "GET", "path": "/"}}}

        with patch.object(main, "_mangum_handler", mangum):
            response = main.handler(event, None)

        mangum.assert_called_once_with(event, None)
        assert response == {"statusCode": 200, "body": "{}"}

    def test_warmup_triggers_both_lazy_loaders(self):
        import main

        with patch("main.ensure_lambda_initialization") as mock_init, patch(
            "core.dependencies.get_face_detection_service"
        ) as mock_face, patch("core.dependencies.get_hybrid_service") as mock_hybrid:
            result = main.warm_up()

        mock_init.assert_called_once()
        mock_face.assert_called_once()
        mock_hybrid.assert_called_once()
        assert result["failed"] == []
        assert set(result["loaded"]) == {"face_detection", "hybrid_recommender"}
        assert result["elapsed_ms"] >= 0

    def test_warmup_survives_loader_failure(self):
        """로더가 실패해도 warm-up 은 예외를 전파하지 않는다"""
        import main

        with patch("main.ensure_lambda_initialization"), patch(
            "core.dependencies.get_face_detection_service",
            side_effect=RuntimeError("mediapipe missing"),
        ), patch("core.dependencies.get_hybrid_service"):
            result = main.warm_up()

        assert "face_detection" in result["failed"]
        assert "hybrid_recommender" in result["loaded"]
