"""
Tests for AWS Secrets Manager integration
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from config.secrets import (
    get_secret,
    is_aws_environment,
    get_secret_or_env
)


class TestSecretsManager:
    """Test AWS Secrets Manager functions"""

    def test_is_aws_environment_lambda(self, monkeypatch):
        """Test AWS environment detection in Lambda"""
        monkeypatch.setenv('AWS_EXECUTION_ENV', 'AWS_Lambda_python3.11')
        assert is_aws_environment() is True

    def test_is_aws_environment_ecs(self, monkeypatch):
        """Test AWS environment detection in ECS"""
        monkeypatch.delenv('AWS_EXECUTION_ENV', raising=False)
        monkeypatch.setenv('ECS_CONTAINER_METADATA_URI', 'http://169.254.170.2/v3')
        assert is_aws_environment() is True

    def test_is_aws_environment_local(self, monkeypatch):
        """Test AWS environment detection in local environment"""
        monkeypatch.delenv('AWS_EXECUTION_ENV', raising=False)
        monkeypatch.delenv('ECS_CONTAINER_METADATA_URI', raising=False)
        assert is_aws_environment() is False

    @patch('config.secrets.boto3')
    def test_get_secret_success(self, mock_boto3):
        """Test successful secret retrieval"""
        # Mock Secrets Manager client
        mock_client = Mock()
        mock_client.get_secret_value.return_value = {
            'SecretString': 'test-secret-value'
        }
        mock_boto3.client.return_value = mock_client

        # Call function
        result = get_secret('hairme-gemini-api-key', 'ap-northeast-2')

        # Verify
        assert result == 'test-secret-value'
        mock_boto3.client.assert_called_once_with('secretsmanager', region_name='ap-northeast-2')
        mock_client.get_secret_value.assert_called_once_with(SecretId='hairme-gemini-api-key')

    @patch('config.secrets.boto3')
    def test_get_secret_not_found(self, mock_boto3):
        """Test secret not found error"""
        from botocore.exceptions import ClientError

        # Mock client error
        mock_client = Mock()
        mock_client.get_secret_value.side_effect = ClientError(
            {'Error': {'Code': 'ResourceNotFoundException', 'Message': 'Secret not found'}},
            'GetSecretValue'
        )
        mock_boto3.client.return_value = mock_client

        # Call function
        result = get_secret('non-existent-secret')

        # Verify
        assert result is None

    @patch('config.secrets.boto3')
    def test_get_secret_access_denied(self, mock_boto3):
        """Test access denied error"""
        from botocore.exceptions import ClientError

        # Mock client error
        mock_client = Mock()
        mock_client.get_secret_value.side_effect = ClientError(
            {'Error': {'Code': 'AccessDeniedException', 'Message': 'Access denied'}},
            'GetSecretValue'
        )
        mock_boto3.client.return_value = mock_client

        # Call function
        result = get_secret('hairme-gemini-api-key')

        # Verify
        assert result is None

    def test_get_secret_boto3_not_available(self):
        """Test when boto3 is not installed"""
        with patch.dict('sys.modules', {'boto3': None}):
            # Clear LRU cache to force re-import
            get_secret.cache_clear()
            result = get_secret('hairme-gemini-api-key')
            assert result is None

    @patch('config.secrets.is_aws_environment')
    @patch('config.secrets.get_secret')
    def test_get_secret_or_env_aws_environment(self, mock_get_secret, mock_is_aws, monkeypatch):
        """Test get_secret_or_env in AWS environment"""
        # Setup
        mock_is_aws.return_value = True
        mock_get_secret.return_value = 'secret-from-aws'

        # Call
        result = get_secret_or_env('hairme-test', 'TEST_ENV_VAR')

        # Verify
        assert result == 'secret-from-aws'
        mock_get_secret.assert_called_once()

    @patch('config.secrets.is_aws_environment')
    def test_get_secret_or_env_local_environment(self, mock_is_aws, monkeypatch):
        """Test get_secret_or_env in local environment"""
        # Setup
        mock_is_aws.return_value = False
        monkeypatch.setenv('TEST_ENV_VAR', 'env-var-value')

        # Call
        result = get_secret_or_env('hairme-test', 'TEST_ENV_VAR', required=False)

        # Verify
        assert result == 'env-var-value'

    @patch('config.secrets.is_aws_environment')
    @patch('config.secrets.get_secret')
    def test_get_secret_or_env_fallback_to_env(self, mock_get_secret, mock_is_aws, monkeypatch):
        """Test fallback to env var when secret fails"""
        # Setup
        mock_is_aws.return_value = True
        mock_get_secret.return_value = None
        monkeypatch.setenv('TEST_ENV_VAR', 'fallback-value')

        # Call
        result = get_secret_or_env('hairme-test', 'TEST_ENV_VAR', required=False)

        # Verify
        assert result == 'fallback-value'

    @patch('config.secrets.is_aws_environment')
    @patch('config.secrets.get_secret')
    def test_get_secret_or_env_required_missing(self, mock_get_secret, mock_is_aws, monkeypatch):
        """Test error when required secret is missing"""
        # Setup
        mock_is_aws.return_value = True
        mock_get_secret.return_value = None
        monkeypatch.delenv('TEST_ENV_VAR', raising=False)

        # Call and verify exception
        with pytest.raises(ValueError, match="Required secret not found"):
            get_secret_or_env('hairme-test', 'TEST_ENV_VAR', required=True)

    @patch('config.secrets.is_aws_environment')
    @patch('config.secrets.get_secret')
    def test_get_secret_or_env_optional_missing(self, mock_get_secret, mock_is_aws, monkeypatch):
        """Test None returned when optional secret is missing"""
        # Setup
        mock_is_aws.return_value = True
        mock_get_secret.return_value = None
        monkeypatch.delenv('TEST_ENV_VAR', raising=False)

        # Call
        result = get_secret_or_env('hairme-test', 'TEST_ENV_VAR', required=False)

        # Verify
        assert result is None


class TestSettingsIntegration:
    """Test Settings class with Secrets Manager integration"""

    @patch('config.settings.is_aws_environment')
    @patch('config.settings.get_secret_or_env')
    def test_settings_loads_from_secrets_manager_in_aws(self, mock_get_secret_or_env, mock_is_aws):
        """Test Settings loads secrets from Secrets Manager in AWS"""
        from config.settings import Settings

        # Setup
        mock_is_aws.return_value = True

        def mock_secret_fetch(secret_name, env_var_name, region_name='ap-northeast-2', required=True):
            secrets = {
                'hairme-gemini-api-key': 'gemini-secret-key',
                'hairme-admin-api-key': 'admin-secret-key',
            }
            return secrets.get(secret_name)

        mock_get_secret_or_env.side_effect = mock_secret_fetch

        # Create settings instance
        settings = Settings()

        # Verify
        assert settings.GEMINI_API_KEY == 'gemini-secret-key'
        assert settings.ADMIN_API_KEY == 'admin-secret-key'

    @patch('config.settings.is_aws_environment')
    def test_settings_uses_env_vars_locally(self, mock_is_aws, monkeypatch):
        """Test Settings uses environment variables in local development"""
        from config.settings import Settings

        # Setup
        mock_is_aws.return_value = False
        monkeypatch.setenv('GEMINI_API_KEY', 'local-gemini-key')
        monkeypatch.setenv('ADMIN_API_KEY', 'local-admin-key')

        # Create settings instance
        settings = Settings()

        # Verify
        assert settings.GEMINI_API_KEY == 'local-gemini-key'
        assert settings.ADMIN_API_KEY == 'local-admin-key'

    @patch('config.settings.is_aws_environment')
    @patch('config.settings.get_secret_or_env')
    def test_settings_raises_error_when_gemini_key_missing(self, mock_get_secret_or_env, mock_is_aws):
        """Test Settings raises error when GEMINI_API_KEY is missing"""
        from config.settings import Settings

        # Setup
        mock_is_aws.return_value = True
        mock_get_secret_or_env.return_value = None

        # Verify error is raised
        with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
            Settings()
