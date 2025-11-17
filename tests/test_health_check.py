"""
Tests for enhanced health check service
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import psutil


class TestHealthCheckService:
    """Test HealthCheckService class"""

    @pytest.fixture
    def health_service(self):
        """Create health check service instance"""
        from core.health_check import HealthCheckService
        return HealthCheckService()

    @pytest.mark.asyncio
    async def test_check_dynamodb_success(self, health_service):
        """Test successful DynamoDB health check"""
        with patch('config.settings.settings') as mock_settings:
            mock_settings.USE_DYNAMODB = True
            mock_settings.AWS_REGION = 'ap-northeast-2'
            mock_settings.DYNAMODB_TABLE_NAME = 'test-table'

            with patch('boto3.resource') as mock_boto:
                # Mock DynamoDB table
                mock_table = Mock()
                mock_table.table_status = 'ACTIVE'

                mock_dynamodb = Mock()
                mock_dynamodb.Table.return_value = mock_table
                mock_boto.return_value = mock_dynamodb

                # Run check
                result = await health_service.check_dynamodb()

                # Verify
                assert result["status"] == "healthy"
                assert result["table_name"] == "test-table"
                assert result["table_status"] == "ACTIVE"
                assert result["latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_check_dynamodb_disabled(self, health_service):
        """Test DynamoDB check when disabled"""
        with patch('config.settings.settings') as mock_settings:
            mock_settings.USE_DYNAMODB = False

            result = await health_service.check_dynamodb()

            assert result["status"] == "skipped"
            assert "MySQL" in result["message"]

    @pytest.mark.asyncio
    async def test_check_dynamodb_error(self, health_service):
        """Test DynamoDB check with connection error"""
        with patch('config.settings.settings') as mock_settings:
            mock_settings.USE_DYNAMODB = True
            mock_settings.AWS_REGION = 'ap-northeast-2'
            mock_settings.DYNAMODB_TABLE_NAME = 'test-table'

            with patch('boto3.resource') as mock_boto:
                from botocore.exceptions import ClientError

                # Mock error
                mock_boto.side_effect = ClientError(
                    {'Error': {'Code': 'ResourceNotFoundException', 'Message': 'Table not found'}},
                    'DescribeTable'
                )

                result = await health_service.check_dynamodb()

                assert result["status"] == "unhealthy"
                assert "error" in result

    @pytest.mark.asyncio
    async def test_check_gemini_api_success(self, health_service):
        """Test successful Gemini API health check"""
        with patch('config.settings.settings') as mock_settings:
            mock_settings.GEMINI_API_KEY = 'test-key'
            mock_settings.MODEL_NAME = 'gemini-1.5-flash-latest'

            with patch('google.generativeai.configure'):
                with patch('google.generativeai.GenerativeModel') as mock_model_class:
                    # Mock model response
                    mock_response = Mock()
                    mock_response.text = "test response"

                    mock_model = Mock()
                    mock_model.generate_content.return_value = mock_response
                    mock_model_class.return_value = mock_model

                    result = await health_service.check_gemini_api()

                    assert result["status"] == "healthy"
                    assert result["model"] == "gemini-1.5-flash-latest"
                    assert result["latency_ms"] > 0
                    assert result["response_length"] > 0

    @pytest.mark.asyncio
    async def test_check_gemini_api_error(self, health_service):
        """Test Gemini API check with API error"""
        with patch('config.settings.settings') as mock_settings:
            mock_settings.GEMINI_API_KEY = 'test-key'
            mock_settings.MODEL_NAME = 'gemini-1.5-flash-latest'

            with patch('google.generativeai.configure'):
                with patch('google.generativeai.GenerativeModel') as mock_model_class:
                    # Mock API error
                    mock_model_class.side_effect = Exception("API Error")

                    result = await health_service.check_gemini_api()

                    assert result["status"] == "unhealthy"
                    assert "error" in result
                    assert "API Error" in result["message"]

    def test_get_system_metrics_success(self, health_service):
        """Test system metrics retrieval"""
        result = health_service.get_system_metrics()

        # Verify structure
        assert "cpu" in result
        assert "memory" in result
        assert "disk" in result

        # Verify CPU metrics
        assert "percent" in result["cpu"]
        assert "count" in result["cpu"]
        assert result["cpu"]["count"] > 0

        # Verify memory metrics
        assert "total_mb" in result["memory"]
        assert "used_mb" in result["memory"]
        assert "percent" in result["memory"]
        assert result["memory"]["total_mb"] > 0

    def test_get_circuit_breaker_status(self, health_service):
        """Test circuit breaker status retrieval"""
        with patch('services.circuit_breaker.gemini_breaker') as mock_breaker:
            mock_breaker.current_state = "closed"
            mock_breaker.fail_counter = 0
            mock_breaker.success_counter = 10

            result = health_service.get_circuit_breaker_status()

            assert result["state"] == "closed"
            assert result["fail_counter"] == 0
            assert result["success_counter"] == 10

    @pytest.mark.asyncio
    async def test_comprehensive_health_check_basic(self, health_service):
        """Test comprehensive health check without deep check"""
        with patch.object(health_service, 'check_dynamodb', new_callable=AsyncMock) as mock_dynamodb:
            with patch.object(health_service, 'get_system_metrics') as mock_system:
                with patch.object(health_service, 'get_circuit_breaker_status') as mock_cb:
                    # Mock successful responses
                    mock_dynamodb.return_value = {"status": "healthy"}
                    mock_system.return_value = {"cpu": {"percent": 50}}
                    mock_cb.return_value = {"state": "closed"}

                    result = await health_service.comprehensive_health_check(
                        include_expensive_checks=False
                    )

                    # Verify structure
                    assert "status" in result
                    assert "timestamp" in result
                    assert "checks" in result
                    assert "check_duration_ms" in result

                    # Verify checks
                    assert "system" in result["checks"]
                    assert "dynamodb" in result["checks"]
                    assert "circuit_breaker" in result["checks"]
                    assert "gemini_api" in result["checks"]

                    # Gemini should be skipped
                    assert result["checks"]["gemini_api"]["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_comprehensive_health_check_deep(self, health_service):
        """Test comprehensive health check with deep check"""
        with patch.object(health_service, 'check_dynamodb', new_callable=AsyncMock) as mock_dynamodb:
            with patch.object(health_service, 'check_gemini_api', new_callable=AsyncMock) as mock_gemini:
                with patch.object(health_service, 'get_system_metrics') as mock_system:
                    with patch.object(health_service, 'get_circuit_breaker_status') as mock_cb:
                        # Mock successful responses
                        mock_dynamodb.return_value = {"status": "healthy"}
                        mock_gemini.return_value = {"status": "healthy", "latency_ms": 100}
                        mock_system.return_value = {"cpu": {"percent": 50}}
                        mock_cb.return_value = {"state": "closed"}

                        result = await health_service.comprehensive_health_check(
                            include_expensive_checks=True
                        )

                        # Gemini should be checked
                        assert result["checks"]["gemini_api"]["status"] == "healthy"

                        # Gemini check should be called
                        mock_gemini.assert_called_once()

    @pytest.mark.asyncio
    async def test_comprehensive_health_check_degraded(self, health_service):
        """Test health check returns degraded when service fails"""
        with patch.object(health_service, 'check_dynamodb', new_callable=AsyncMock) as mock_dynamodb:
            with patch.object(health_service, 'get_system_metrics') as mock_system:
                with patch.object(health_service, 'get_circuit_breaker_status') as mock_cb:
                    # Mock DynamoDB failure
                    mock_dynamodb.return_value = {"status": "unhealthy", "error": "Connection failed"}
                    mock_system.return_value = {"cpu": {"percent": 50}}
                    mock_cb.return_value = {"state": "closed"}

                    result = await health_service.comprehensive_health_check()

                    # Status should be degraded
                    assert result["status"] == "degraded"


class TestHealthCheckEndpoint:
    """Test /api/health endpoint"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        with patch('config.settings.is_aws_environment', return_value=False):
            with patch.dict('os.environ', {
                'GEMINI_API_KEY': 'test-key',
                'ENVIRONMENT': 'development'
            }):
                from fastapi.testclient import TestClient
                from main import app
                return TestClient(app)

    def test_health_endpoint_basic(self, client):
        """Test basic health check endpoint"""
        with patch('core.health_check.get_health_check_service') as mock_service:
            mock_health_service = Mock()
            mock_health_service.comprehensive_health_check = AsyncMock(return_value={
                "status": "healthy",
                "timestamp": "2025-01-17T00:00:00",
                "checks": {
                    "system": {"cpu": {"percent": 50}},
                    "dynamodb": {"status": "healthy"},
                    "circuit_breaker": {"state": "closed"},
                    "gemini_api": {"status": "skipped"}
                },
                "check_duration_ms": 50
            })
            mock_service.return_value = mock_health_service

            response = client.get("/api/health")

            assert response.status_code == 200
            data = response.json()

            assert "status" in data
            assert "version" in data
            assert "checks" in data
            assert "startup" in data

    def test_health_endpoint_deep_check(self, client):
        """Test health check with deep=true"""
        with patch('core.health_check.get_health_check_service') as mock_service:
            mock_health_service = Mock()
            mock_health_service.comprehensive_health_check = AsyncMock(return_value={
                "status": "healthy",
                "timestamp": "2025-01-17T00:00:00",
                "checks": {
                    "system": {"cpu": {"percent": 50}},
                    "dynamodb": {"status": "healthy"},
                    "circuit_breaker": {"state": "closed"},
                    "gemini_api": {"status": "healthy", "latency_ms": 100}
                },
                "check_duration_ms": 150
            })
            mock_service.return_value = mock_health_service

            response = client.get("/api/health?deep=true")

            assert response.status_code == 200
            data = response.json()

            # Verify deep check was called
            mock_health_service.comprehensive_health_check.assert_called_once_with(
                include_expensive_checks=True
            )
