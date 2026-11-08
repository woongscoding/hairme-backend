"""
Tests for improved error handling with specific exceptions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestMLRecommenderErrorHandling:
    """Test error handling in MLRecommendationService (formerly HybridRecommender)"""

    @pytest.fixture
    def ml_service(self):
        """Create ML recommendation service with mocked components"""
        mock_recommender = Mock()
        mock_recommender.recommend_top_k.return_value = []

        mock_reason_gen = Mock()

        with patch(
            "models.ml_recommender.get_ml_recommender",
            return_value=mock_recommender,
        ):
            with patch(
                "services.reason_generator.get_reason_generator",
                return_value=mock_reason_gen,
            ):
                from services.hybrid_recommender import MLRecommendationService

                service = MLRecommendationService()
                return service

    def test_ml_recommend_handles_ml_failure_gracefully(self, ml_service):
        """Test that ML recommendation failure is handled gracefully"""
        ml_service.ml_recommender.recommend_top_k.side_effect = Exception(
            "ML model error"
        )

        # Should not raise - logs error and returns empty recommendations
        with patch(
            "services.trending_style_service.get_trending_style_service",
            side_effect=Exception("no trending"),
        ):
            result = ml_service.recommend(b"test_image", "계란형", "가을웜")

        assert "analysis" in result
        assert "recommendations" in result
        assert result["analysis"]["face_shape"] == "계란형"

    def test_ml_recommend_returns_results_on_success(self, ml_service):
        """Test that ML recommendation returns results on success"""
        ml_service.ml_recommender.recommend_top_k.return_value = [
            {"hairstyle_id": 1, "hairstyle": "레이어드 컷", "score": 85.0},
        ]
        ml_service.reason_generator.generate_with_score.return_value = (
            "얼굴형에 잘 어울립니다"
        )

        with patch(
            "services.trending_style_service.get_trending_style_service",
            side_effect=Exception("no trending"),
        ):
            result = ml_service.recommend(b"test_image", "계란형", "가을웜")

        assert len(result["recommendations"]) == 1
        assert result["recommendations"][0]["style_name"] == "레이어드 컷"

    def test_ml_service_init_fails_when_recommender_unavailable(self):
        """Test that MLRecommendationService raises when ML recommender is unavailable"""
        with patch(
            "models.ml_recommender.get_ml_recommender",
            side_effect=Exception("Model not found"),
        ):
            from services.hybrid_recommender import MLRecommendationService

            with pytest.raises(Exception, match="Model not found"):
                MLRecommendationService()


class TestHealthCheckErrorHandling:
    """Test health check error handling"""

    @pytest.mark.asyncio
    async def test_dynamodb_check_handles_client_error(self):
        """Test that DynamoDB check handles ClientError properly"""
        from core.health_check import HealthCheckService

        health_service = HealthCheckService()

        with patch("config.settings.settings") as mock_settings:
            mock_settings.USE_DYNAMODB = True
            mock_settings.AWS_REGION = "ap-northeast-2"
            mock_settings.DYNAMODB_TABLE_NAME = "test-table"

            with patch("boto3.resource") as mock_boto:
                from botocore.exceptions import ClientError

                mock_boto.side_effect = ClientError(
                    {
                        "Error": {
                            "Code": "ResourceNotFoundException",
                            "Message": "Table not found",
                        }
                    },
                    "DescribeTable",
                )

                result = await health_service.check_dynamodb()

                assert result["status"] == "unhealthy"
                assert result["latency_ms"] == 0

    @pytest.mark.asyncio
    async def test_gemini_check_handles_api_error(self):
        """Test that Gemini check handles API errors properly"""
        from core.health_check import HealthCheckService

        health_service = HealthCheckService()

        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value.models.get.side_effect = Exception("API Error")

            result = await health_service.check_gemini_api()

            assert result["status"] == "unhealthy"
            assert result["latency_ms"] == 0

    def test_system_metrics_handles_disk_permission_error(self):
        """Test that system metrics handles disk permission errors"""
        from core.health_check import HealthCheckService

        health_service = HealthCheckService()

        with patch("psutil.disk_usage") as mock_disk:
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

        with patch("boto3.client") as mock_boto:
            from botocore.exceptions import ClientError

            mock_client = Mock()
            mock_client.get_secret_value.side_effect = ClientError(
                {"Error": {"Code": "AccessDenied Error", "Message": "Access denied"}},
                "GetSecretValue",
            )
            mock_boto.return_value = mock_client

            result = get_secret("test-secret")

            # Should return None, not raise
            assert result is None
