"""
AWS Secrets Manager integration for secure credential management

This module provides caching and fallback mechanisms for retrieving secrets.
"""

import os
import logging
from typing import Dict, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def get_secret(secret_name: str, region_name: str = "ap-northeast-2") -> Optional[str]:
    """
    Retrieve a secret from AWS Secrets Manager with caching

    Args:
        secret_name: Name of the secret in Secrets Manager
        region_name: AWS region (default: ap-northeast-2)

    Returns:
        Secret value as string, or None if retrieval fails

    Note:
        - Results are cached using @lru_cache to avoid repeated API calls
        - Only works in AWS environments (Lambda, EC2, ECS)
        - Falls back to None if boto3 is unavailable or secret not found
    """
    try:
        import boto3
        from botocore.exceptions import ClientError

        # Create Secrets Manager client
        client = boto3.client('secretsmanager', region_name=region_name)

        # Retrieve secret
        response = client.get_secret_value(SecretId=secret_name)

        # Extract secret string
        secret_value: Optional[str] = response.get('SecretString')

        logger.info(f"✅ Successfully retrieved secret: {secret_name}")
        return secret_value

    except ClientError as e:
        error_code = e.response['Error']['Code']

        if error_code == 'ResourceNotFoundException':
            logger.warning(f"⚠️ Secret not found: {secret_name}")
        elif error_code == 'AccessDeniedException':
            logger.error(f"❌ Access denied to secret: {secret_name}")
        elif error_code == 'InvalidRequestException':
            logger.error(f"❌ Invalid request for secret: {secret_name}")
        else:
            logger.error(f"❌ Error retrieving secret {secret_name}: {e}")

        return None

    except ImportError:
        logger.warning("⚠️ boto3 not available - cannot retrieve secrets")
        return None

    except Exception as e:
        logger.error(f"❌ Unexpected error retrieving secret {secret_name}: {str(e)}")
        return None


def is_aws_environment() -> bool:
    """
    Check if running in AWS environment (Lambda, EC2, ECS, etc.)

    Returns:
        True if in AWS environment, False otherwise
    """
    # Check for Lambda environment
    if os.getenv('AWS_EXECUTION_ENV'):
        return True

    # Check for ECS environment
    if os.getenv('ECS_CONTAINER_METADATA_URI'):
        return True

    # Check for EC2 instance metadata (this is more expensive, so check last)
    try:
        import requests  # type: ignore[import-untyped]
        # EC2 metadata service with short timeout
        response = requests.get(
            'http://169.254.169.254/latest/meta-data/',
            timeout=0.1
        )
        return bool(response.status_code == 200)
    except (ImportError, ModuleNotFoundError):
        # requests not installed
        pass
    except (ConnectionError, TimeoutError):
        # Not on EC2 or metadata service unavailable
        pass
    except Exception:
        # Any other error (don't log, this is expected in non-EC2 environments)
        pass

    return False


def get_secret_or_env(
    secret_name: str,
    env_var_name: str,
    region_name: str = "ap-northeast-2",
    required: bool = True
) -> Optional[str]:
    """
    Get value from Secrets Manager if in AWS environment, otherwise from environment variable

    Priority:
    1. AWS Secrets Manager (if in AWS environment)
    2. Environment variable (fallback)
    3. None (if required=False)
    4. Raise error (if required=True and not found)

    Args:
        secret_name: Name of the secret in Secrets Manager
        env_var_name: Name of the environment variable
        region_name: AWS region
        required: Whether the value is required (raise error if missing)

    Returns:
        Secret value or None

    Raises:
        ValueError: If required=True and value not found

    Example:
        >>> api_key = get_secret_or_env('hairme-gemini-api-key', 'GEMINI_API_KEY')
    """
    # Try Secrets Manager first if in AWS
    if is_aws_environment():
        logger.info(f"🔐 AWS environment detected - retrieving secret: {secret_name}")
        secret_value = get_secret(secret_name, region_name)

        if secret_value:
            logger.info(f"✅ Using secret from Secrets Manager: {secret_name}")
            return secret_value
        else:
            logger.warning(f"⚠️ Failed to retrieve secret {secret_name}, falling back to env var")

    # Fallback to environment variable
    env_value = os.getenv(env_var_name)

    if env_value:
        logger.info(f"✅ Using environment variable: {env_var_name}")
        return env_value

    # Handle missing value
    if required:
        error_msg = (
            f"Required secret not found: {secret_name} (Secrets Manager) "
            f"or {env_var_name} (environment variable)"
        )
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)

    logger.warning(f"⚠️ Optional secret not found: {secret_name}/{env_var_name}")
    return None
