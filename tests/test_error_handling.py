"""
Tests for improved error handling with specific exceptions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from core.exceptions import (
    GeminiAPIException,
    GeminiRateLimitException,
    GeminiInvalidResponseException,
    GeminiAuthenticationException,
    StyleNotFoundException,
    CircuitBreakerOpenException,
    MediaPipeException,
    DynamoDBException
)


class TestExceptionHierarchy:
    """Test exception class hierarchy"""

    def test_gemini_rate_limit_is_gemini_exception(self):
        """Test that GeminiRateLimitException is subclass of GeminiAPIException"""
        exc = GeminiRateLimitException()
        assert isinstance(exc, GeminiAPIException)

    def test_gemini_invalid_response_is_gemini_exception(self):
        """Test that GeminiInvalidResponseException is subclass of GeminiAPIException"""
        exc = GeminiInvalidResponseException()
        assert isinstance(exc, GeminiAPIException)

    def test_style_not_found_is_ml_exception(self):
        """Test that StyleNotFoundException is subclass of MLModelException"""
        from core.exceptions import MLModelException
        exc = StyleNotFoundException("test-style")
        assert isinstance(exc, MLModelException)

    def test_exception_messages(self):
        """Test exception default messages"""
        assert "rate limit" in GeminiRateLimitException().message.lower() or "한도" in GeminiRateLimitException().message
        assert "파싱" in GeminiInvalidResponseException().message
        assert "인증" in GeminiAuthenticationException().message


class TestHybridRecommenderErrorHandling:
    """Test error handling in HybridRecommender"""

    @pytest.fixture
    def hybrid_service(self):
        """Create hybrid service with mocked components"""
        with patch('services.hybrid_recommender.get_ml_recommender') as mock_ml:
            with patch('services.hybrid_recommender.get_reason_generator') as mock_reason:
                mock_ml.return_value = None
                mock_reason.return_value = None

                from services.hybrid_recommender import HybridRecommendationService
                service = HybridRecommendationService(gemini_api_key="test-key")
                return service

    @pytest.mark.asyncio
    async def test_gemini_json_parse_error_raises_specific_exception(self, hybrid_service):
        """Test that JSON parse errors raise GeminiInvalidResponseException"""
        with patch.object(hybrid_service.gemini_model, 'generate_content') as mock_generate:
            # Mock response with invalid JSON
            mock_response = Mock()
            mock_response.text = "This is not JSON"
            mock_generate.return_value = mock_response

            with pytest.raises(GeminiInvalidResponseException):
                await hybrid_service._call_gemini(b"test_image", "계란형", "가을웜")

    @pytest.mark.asyncio
    async def test_gemini_rate_limit_detection(self, hybrid_service):
        """Test that rate limit errors are detected and raise specific exception"""
        with patch.object(hybrid_service.gemini_model, 'generate_content') as mock_generate:
            # Mock rate limit error
            mock_generate.side_effect = Exception("quota exceeded for this resource")

            with pytest.raises(GeminiRateLimitException):
                await hybrid_service._call_gemini(b"test_image", "계란형", "가을웜")

    @pytest.mark.asyncio
    async def test_gemini_auth_error_detection(self, hybrid_service):
        """Test that authentication errors are detected"""
        with patch.object(hybrid_service.gemini_model, 'generate_content') as mock_generate:
            # Mock auth error
            mock_generate.side_effect = Exception("API key invalid")

            with pytest.raises(GeminiAuthenticationException):
                await hybrid_service._call_gemini(b"test_image", "계란형", "가을웜")

    @pytest.mark.asyncio
    async def test_gemini_connection_error_detection(self, hybrid_service):
        """Test that connection errors are detected"""
        with patch.object(hybrid_service.gemini_model, 'generate_content') as mock_generate:
            # Mock connection error
            mock_generate.side_effect = Exception("connection timeout")

            with pytest.raises(GeminiAPIException) as exc_info:
                await hybrid_service._call_gemini(b"test_image", "계란형", "가을웜")

            assert "연결 실패" in str(exc_info.value)

    def test_ml_score_prediction_handles_key_error(self, hybrid_service):
        """Test that KeyError in ML prediction is properly handled"""
        # Enable ML
        mock_recommender = Mock()
        mock_recommender.predict_score.side_effect = KeyError("style not found")
        hybrid_service.ml_recommender = mock_recommender
        hybrid_service.ml_available = True

        # This should not raise, but log warning and return 0.0
        gemini_recs = [{"style_name": "테스트 스타일", "reason": "테스트"}]
        ml_recs = []

        result = hybrid_service._merge_recommendations(
            gemini_recs, ml_recs, "계란형", "가을웜"
        )

        # Should still return recommendations
        assert len(result) > 0
        assert result[0]["score"] == 0.0  # Default score on error

    def test_ml_score_prediction_handles_value_error(self, hybrid_service):
        """Test that ValueError in ML prediction is properly handled"""
        mock_recommender = Mock()
        mock_recommender.predict_score.side_effect = ValueError("invalid input")
        hybrid_service.ml_recommender = mock_recommender
        hybrid_service.ml_available = True

        gemini_recs = [{"style_name": "테스트 스타일", "reason": "테스트"}]
        ml_recs = []

        result = hybrid_service._merge_recommendations(
            gemini_recs, ml_recs, "계란형", "가을웜"
        )

        assert len(result) > 0
        assert result[0]["score"] == 0.0


class TestCircuitBreakerErrorHandling:
    """Test circuit breaker error handling"""

    def test_circuit_breaker_open_raises_custom_exception(self):
        """Test that CircuitBreakerError without fallback raises CircuitBreakerOpenException"""
        from pybreaker import CircuitBreaker, CircuitBreakerError
        from services.circuit_breaker import with_circuit_breaker

        # Create test breaker
        test_breaker = CircuitBreaker(fail_max=1, timeout_duration=60, name="TestService")

        # Function that always fails
        @with_circuit_breaker(test_breaker, fallback=None)
        def failing_function():
            raise Exception("Test error")

        # First call should fail
        with pytest.raises(Exception):
            failing_function()

        # Second call should open circuit and raise CircuitBreakerOpenException
        with pytest.raises(CircuitBreakerOpenException) as exc_info:
            failing_function()

        assert "TestService" in str(exc_info.value)

    def test_circuit_breaker_with_fallback_uses_fallback(self):
        """Test that circuit breaker uses fallback when open"""
        from pybreaker import CircuitBreaker
        from services.circuit_breaker import with_circuit_breaker

        test_breaker = CircuitBreaker(fail_max=1, timeout_duration=60, name="TestService")

        def fallback_function(*args, **kwargs):
            return {"fallback": True}

        @with_circuit_breaker(test_breaker, fallback=fallback_function)
        def failing_function():
            raise Exception("Test error")

        # Fail once to open circuit
        with pytest.raises(Exception):
            failing_function()

        # Next call should use fallback
        result = failing_function()
        assert result["fallback"] is True


class TestHealthCheckErrorHandling:
    """Test health check error handling"""

    @pytest.mark.asyncio
    async def test_dynamodb_check_handles_client_error(self):
        """Test that DynamoDB check handles ClientError properly"""
        from core.health_check import HealthCheckService

        health_service = HealthCheckService()

        with patch('config.settings.settings') as mock_settings:
            mock_settings.USE_DYNAMODB = True
            mock_settings.AWS_REGION = 'ap-northeast-2'
            mock_settings.DYNAMODB_TABLE_NAME = 'test-table'

            with patch('boto3.resource') as mock_boto:
                from botocore.exceptions import ClientError

                mock_boto.side_effect = ClientError(
                    {'Error': {'Code': 'ResourceNotFoundException', 'Message': 'Table not found'}},
                    'DescribeTable'
                )

                result = await health_service.check_dynamodb()

                assert result["status"] == "unhealthy"
                assert "ResourceNotFoundException" in result["error"]

    @pytest.mark.asyncio
    async def test_gemini_check_handles_api_error(self):
        """Test that Gemini check handles API errors properly"""
        from core.health_check import HealthCheckService

        health_service = HealthCheckService()

        with patch('config.settings.settings') as mock_settings:
            mock_settings.GEMINI_API_KEY = 'test-key'
            mock_settings.MODEL_NAME = 'gemini-1.5-flash-latest'

            with patch('google.generativeai.configure'):
                with patch('google.generativeai.GenerativeModel') as mock_model_class:
                    mock_model_class.side_effect = Exception("API Error")

                    result = await health_service.check_gemini_api()

                    assert result["status"] == "unhealthy"
                    assert "error" in result
                    assert "API Error" in result["message"]

    def test_system_metrics_handles_disk_permission_error(self):
        """Test that system metrics handles disk permission errors"""
        from core.health_check import HealthCheckService

        health_service = HealthCheckService()

        with patch('psutil.disk_usage') as mock_disk:
            mock_disk.side_effect = PermissionError("Access denied")

            result = health_service.get_system_metrics()

            # Should still return CPU and memory
            assert "cpu" in result
            assert "memory" in result

            # Disk should indicate unavailable
            assert result["disk"]["status"] == "unavailable"


class TestSecretsErrorHandling:
    """Test secrets manager error handling"""

    def test_is_aws_environment_handles_missing_requests(self):
        """Test that is_aws_environment handles missing requests module"""
        from config.secrets import is_aws_environment

        # Should not raise even if requests is not available
        # Just returns False for EC2 check
        result = is_aws_environment()

        # Should be False in test environment
        assert result is False

    def test_get_secret_handles_client_error(self):
        """Test that get_secret handles ClientError properly"""
        from config.secrets import get_secret

        with patch('boto3.client') as mock_boto:
            from botocore.exceptions import ClientError

            mock_client = Mock()
            mock_client.get_secret_value.side_effect = ClientError(
                {'Error': {'Code': 'AccessDenied Error', 'Message': 'Access denied'}},
                'GetSecretValue'
            )
            mock_boto.return_value = mock_client

            result = get_secret('test-secret')

            # Should return None, not raise
            assert result is None
