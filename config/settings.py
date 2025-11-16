"""Application settings and configuration management using Pydantic Settings"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables"""

    # API Keys
    GEMINI_API_KEY: str

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
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]
    API_KEY: Optional[str] = None

    # Gemini Model Configuration
    MODEL_NAME: str = "gemini-flash-latest"

    # ML Model Paths
    ML_MODEL_PATH: str = "models/final_model.pth"
    ML_ENCODER_PATH: str = "models/encoders.pkl"

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


# Singleton instance
settings = Settings()
