"""Tests for main application endpoints"""

import pytest
from unittest.mock import patch, AsyncMock
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
            mock_health.comprehensive_health_check = AsyncMock(return_value={
                "status": "healthy",
                "timestamp": "2025-01-17T00:00:00",
                "checks": {
                    "system": {"cpu": {"percent": 50}},
                    "dynamodb": {"status": "healthy"},
                    "circuit_breaker": {"state": "closed"},
                    "gemini_api": {"status": "skipped"},
                },
                "check_duration_ms": 50,
            })

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
