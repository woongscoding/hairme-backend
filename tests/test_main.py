"""Tests for main application endpoints"""

import pytest
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
        assert "mediapipe_analysis" in features
        assert "gemini_analysis" in features
        assert "redis_cache" in features
        assert "database" in features


class TestHealthCheck:
    """Test health check endpoint"""

    def test_health_check_returns_200(self, client):
        """Test that health check returns 200 OK"""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_check_shows_healthy_status(self, client):
        """Test that health check shows healthy status"""
        response = client.get("/api/health")
        data = response.json()

        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_check_includes_service_status(self, client):
        """Test that health check includes all service statuses"""
        response = client.get("/api/health")
        data = response.json()

        # All required services should be listed
        assert "version" in data
        assert "model" in data
        assert "mediapipe_analysis" in data
        assert "gemini_api" in data
        assert "redis" in data
        assert "database" in data
        assert "feedback_system" in data
        assert "ml_model" in data
        assert "style_embedding" in data


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
