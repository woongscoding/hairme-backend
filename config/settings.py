"""Application settings and configuration management using Pydantic Settings"""

import os
import logging
from typing import Any, List, Optional
from pydantic_settings import BaseSettings
from config.secrets import get_secret_or_env, is_aws_environment

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application configuration loaded from environment variables"""

    # API Keys (will be overridden by __init__ if in AWS)
    GEMINI_API_KEY: str = ""
    ADMIN_API_KEY: Optional[str] = None  # For admin endpoints authentication

    # Database Configuration
    DATABASE_URL: Optional[str] = None
    DB_PASSWORD: Optional[str] = None

    # DynamoDB Configuration
    USE_DYNAMODB: bool = False
    AWS_REGION: str = "ap-northeast-2"
    DYNAMODB_TABLE_NAME: str = "hairme-analysis"

    # Redis Configuration
    REDIS_URL: Optional[str] = None
    CACHE_TTL: int = 86400  # 24 hours in seconds

    # Security Settings
    ALLOWED_ORIGINS: str = "http://localhost:3000"  # Comma-separated list
    API_KEY: Optional[str] = None

    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse ALLOWED_ORIGINS string into list"""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    # Gemini Model Configuration
    MODEL_NAME: str = "gemini-flash-latest"

    # ML Model Paths
    ML_MODEL_PATH: str = "models/final_model.pth"
    ML_ENCODER_PATH: str = "models/encoders.pkl"

    # ML Model Version Management
    MODEL_ENVIRONMENT: str = "production"  # production, staging, or archive/vX_YYYY-MM-DD
    MODEL_BASE_PATH: str = "models"

    @property
    def active_model_path(self) -> str:
        """Get the active model path based on MODEL_ENVIRONMENT"""
        if self.MODEL_ENVIRONMENT in ["production", "staging"]:
            return f"{self.MODEL_BASE_PATH}/{self.MODEL_ENVIRONMENT}/model.pth"
        elif self.MODEL_ENVIRONMENT.startswith("archive/"):
            return f"{self.MODEL_BASE_PATH}/{self.MODEL_ENVIRONMENT}/model.pt"
        else:
            # Fallback to legacy path
            return self.ML_MODEL_PATH

    @property
    def active_encoder_path(self) -> str:
        """Get the active encoder path based on MODEL_ENVIRONMENT"""
        if self.MODEL_ENVIRONMENT in ["production", "staging"]:
            return f"{self.MODEL_BASE_PATH}/{self.MODEL_ENVIRONMENT}/encoders.pkl"
        else:
            # Fallback to legacy path
            return self.ML_ENCODER_PATH

    @property
    def model_metadata_path(self) -> Optional[str]:
        """Get the metadata file path for the current model environment"""
        if self.MODEL_ENVIRONMENT in ["production", "staging"]:
            return f"{self.MODEL_BASE_PATH}/{self.MODEL_ENVIRONMENT}/metadata.json"
        elif self.MODEL_ENVIRONMENT.startswith("archive/"):
            return f"{self.MODEL_BASE_PATH}/{self.MODEL_ENVIRONMENT}/metadata.json"
        else:
            return None

    # Sentence Transformer Configuration
    SENTENCE_TRANSFORMER_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # Logging Configuration
    LOG_LEVEL: str = "INFO"

    # Environment Settings
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # Application Info
    APP_TITLE: str = "HairMe API"
    APP_DESCRIPTION: str = "AI 기반 헤어스타일 추천 서비스 (v20.2.0: MediaPipe 전환 완료)"
    APP_VERSION: str = "20.2.0"

    # Constants
    CONFIDENCE_THRESHOLD_VERY_HIGH: float = 0.90
    CONFIDENCE_THRESHOLD_HIGH: float = 0.85
    CONFIDENCE_THRESHOLD_MEDIUM: float = 0.75

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize settings with AWS Secrets Manager integration

        Priority:
        1. AWS Secrets Manager (if in AWS environment)
        2. Environment variables (fallback)
        3. .env file (fallback)
        """
        super().__init__(**kwargs)

        # Only fetch from Secrets Manager if in AWS environment
        if is_aws_environment():
            logger.info("🔐 AWS environment detected - loading secrets from Secrets Manager")

            # Fetch GEMINI_API_KEY from Secrets Manager
            try:
                gemini_key = get_secret_or_env(
                    secret_name='hairme-gemini-api-key',
                    env_var_name='GEMINI_API_KEY',
                    region_name=self.AWS_REGION,
                    required=True
                )
                if gemini_key:
                    self.GEMINI_API_KEY = gemini_key
                    logger.info("✅ GEMINI_API_KEY loaded from Secrets Manager")
            except Exception as e:
                logger.error(f"❌ Failed to load GEMINI_API_KEY: {str(e)}")

            # Fetch ADMIN_API_KEY from Secrets Manager
            try:
                admin_key = get_secret_or_env(
                    secret_name='hairme-admin-api-key',
                    env_var_name='ADMIN_API_KEY',
                    region_name=self.AWS_REGION,
                    required=False
                )
                if admin_key:
                    self.ADMIN_API_KEY = admin_key
                    logger.info("✅ ADMIN_API_KEY loaded from Secrets Manager")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load ADMIN_API_KEY: {str(e)}")

            # Fetch DB_PASSWORD from Secrets Manager (if using MySQL)
            if not self.USE_DYNAMODB:
                try:
                    db_password = get_secret_or_env(
                        secret_name='hairme-db-password',
                        env_var_name='DB_PASSWORD',
                        region_name=self.AWS_REGION,
                        required=False
                    )
                    if db_password:
                        self.DB_PASSWORD = db_password
                        logger.info("✅ DB_PASSWORD loaded from Secrets Manager")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load DB_PASSWORD: {str(e)}")

            # Fetch DATABASE_URL from Secrets Manager (if using MySQL)
            if not self.USE_DYNAMODB and not self.DATABASE_URL:
                try:
                    database_url = get_secret_or_env(
                        secret_name='hairme-database-url',
                        env_var_name='DATABASE_URL',
                        region_name=self.AWS_REGION,
                        required=False
                    )
                    if database_url:
                        self.DATABASE_URL = database_url
                        logger.info("✅ DATABASE_URL loaded from Secrets Manager")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load DATABASE_URL: {str(e)}")

        else:
            logger.info("💻 Local/Dev environment detected - using environment variables/.env file")

        # Validate required secrets
        if not self.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is required but not found in Secrets Manager or environment variables"
            )


# Singleton instance
settings = Settings()
