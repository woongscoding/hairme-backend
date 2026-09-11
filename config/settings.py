"""Application settings and configuration management using Pydantic Settings"""

import os
import logging
from typing import Any, Dict, List, Optional
from pydantic_settings import BaseSettings
from config.secrets import get_secret_or_env, is_aws_environment

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application configuration loaded from environment variables"""

    # API Keys (will be overridden by __init__ if in AWS)
    GEMINI_API_KEY: str = ""
    ADMIN_API_KEY: Optional[str] = None  # For admin endpoints authentication
    OPENAI_API_KEY: Optional[str] = None  # LangGraph 챗봇 (임베딩 + LLM)
    TAVILY_API_KEY: Optional[str] = None  # 챗봇 web_search 노드

    # Database Configuration (legacy MySQL - 더 이상 사용되지 않음, 하위 호환용 필드)
    DATABASE_URL: Optional[str] = None
    DB_PASSWORD: Optional[str] = None

    # DynamoDB Configuration
    # USE_DYNAMODB: 배포 호환용 플래그. DynamoDB가 유일한 백엔드이므로
    # 더 이상 백엔드를 선택하지 않는다 (dynamodb_connection.init_dynamodb 게이트에만 사용).
    USE_DYNAMODB: bool = False
    AWS_REGION: str = "ap-northeast-2"
    DYNAMODB_TABLE_NAME: str = "hairme-analysis"
    DYNAMODB_USAGE_TABLE_NAME: str = "hairstyle_usage"

    # Daily Synthesis Limit
    DAILY_SYNTHESIS_LIMIT: int = 3
    # 비로그인(device_id) 합성은 device_id 회전으로 우회 가능하므로 IP 단위 일일 상한을 추가로 둔다.
    ANON_IP_DAILY_SYNTHESIS_LIMIT: int = 15

    # ===== 회원 인증 (Kakao 로그인 + 자체 JWT) =====
    # 우리 카카오 앱의 네이티브/REST 앱 ID (숫자). 설정 시 access_token_info로
    # 토큰이 우리 앱에서 발급된 것인지 검증한다 (토큰 치환 공격 차단).
    # 미설정 시 검증 생략 (기존 배포 호환) - 프로덕션에서는 반드시 설정할 것.
    KAKAO_APP_ID: str = ""
    JWT_SECRET_KEY: str = (
        ""  # 프로덕션에서는 Secrets Manager(hairme-jwt-secret)에서 로드
    )
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # DynamoDB 사용자/크레딧 테이블
    DYNAMODB_USERS_TABLE_NAME: str = "hairme-users"
    DYNAMODB_CREDIT_LEDGER_TABLE_NAME: str = "hairme-credit-ledger"

    # ===== 레거시 비로그인(device_id) 합성 흐름 =====
    # False 로 내리면 비로그인 합성은 401(로그인 필요)로 거부한다. 구버전 앱 사용 비율을
    # 로그(event_type=legacy_device_flow)로 확인한 뒤 내릴 것.
    LEGACY_DEVICE_FLOW_ENABLED: bool = True

    # ===== 크레딧 정책 =====
    SIGNUP_BONUS_CREDITS: int = 5  # 가입 보너스 (평생 무료분)
    SYNTHESIS_CREDIT_COST: int = 1  # 합성 1회당 차감 크레딧

    # ===== 사진 저장 (S3) =====
    PHOTO_S3_BUCKET: str = ""  # 비어있으면 사진 저장/결과 캐싱 비활성화
    PHOTO_URL_EXPIRE_SECONDS: int = 86400  # presigned URL 유효기간 (24시간)

    # ===== Google Play 인앱결제 =====
    PLAY_PACKAGE_NAME: str = (
        ""  # 앱 패키지명 (예: com.hairme.app), 미설정 시 구매 검증 비활성화
    )
    PLAY_SERVICE_ACCOUNT_JSON: str = (
        ""  # 서비스 계정 키 JSON (프로덕션: Secrets Manager hairme-play-service-account)
    )
    # 스토어 상품 ID → 지급 크레딧 매핑 (환경변수로 덮어쓸 땐 JSON 문자열)
    CREDIT_PRODUCTS: Dict[str, int] = {
        "credits_10": 10,
        "credits_30": 30,
        "credits_100": 100,
    }

    # ===== Google Play 환불/취소 회수 (voidedpurchases 폴링) =====
    PLAY_VOID_RECLAIM_ENABLED: bool = (
        True  # 잡 이벤트 {"job":"reclaim_voided_purchases"} 처리 여부
    )
    PLAY_VOID_LOOKBACK_DAYS: int = 30  # 폴링 시 조회할 과거 기간 (Play 최대 30일)

    # ===== AdMob 리워드 광고 (SSV) =====
    REWARD_AD_DAILY_LIMIT: int = 5  # 유저당 하루 보상 횟수 상한
    # 우리 앱의 AdMob 리워드 광고 단위 ID 허용목록 (쉼표 구분).
    # SSV 검증 키는 전 퍼블리셔 공용이므로, 비어 있으면 프로덕션에서 콜백을 거부한다.
    ADMOB_REWARD_AD_UNIT_IDS: str = ""

    # ===== 제휴 커머스 (쿠팡파트너스) =====
    # 클릭 로그 테이블 (PK: user_id, SK: sk) - 콘솔에서 생성 필요
    DYNAMODB_AFFILIATE_CLICKS_TABLE_NAME: str = "hairme-affiliate-clicks"

    # MLOps Configuration
    MLOPS_ENABLED: bool = False  # MLOps 파이프라인 활성화
    MLOPS_S3_BUCKET: str = "hairme-mlops"  # MLOps S3 버킷
    MLOPS_RETRAIN_THRESHOLD: int = 100  # 재학습 트리거 피드백 수
    MLOPS_TRAINER_LAMBDA: str = "hairme-model-trainer"  # Trainer Lambda 함수명
    MLOPS_SNS_TOPIC_ARN: str = ""  # 알림용 SNS 토픽 (선택)

    # Redis Configuration
    REDIS_URL: Optional[str] = None
    CACHE_TTL: int = 86400  # 24 hours in seconds

    # Security Settings
    ALLOWED_ORIGINS: str = "http://localhost:3000"  # Comma-separated list
    API_KEY: Optional[str] = None

    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse ALLOWED_ORIGINS string into list"""
        return [
            origin.strip()
            for origin in self.ALLOWED_ORIGINS.split(",")
            if origin.strip()
        ]

    # Gemini Model Configuration
    MODEL_NAME: str = "gemini-2.5-flash"
    # 이미지 생성(합성/염색) 전용 모델.
    # gemini-2.5-flash-image 는 2026-10-02 셧다운 예정이며,
    # gemini-3.1-flash-image 는 동일한 지연시간으로 측정되어 기본값으로 승격했다.
    GEMINI_IMAGE_MODEL: str = "gemini-3.1-flash-image"

    # ML Model Paths
    ML_MODEL_PATH: str = "models/final_model.pth"
    ML_ENCODER_PATH: str = "models/encoders.pkl"

    # ML Model Version Management
    MODEL_ENVIRONMENT: str = (
        "production"  # production, staging, or archive/vX_YYYY-MM-DD
    )
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
    APP_TITLE: str = "BeautyMe API"
    APP_DESCRIPTION: str = (
        "AI 기반 종합 뷰티 컨설팅 플랫폼 (v23.0.0: 얼굴분석 + 퍼스널컬러 + 헤어추천 + AI합성)"
    )
    APP_VERSION: str = "23.0.0"

    # Constants
    CONFIDENCE_THRESHOLD_VERY_HIGH: float = 0.90
    CONFIDENCE_THRESHOLD_HIGH: float = 0.85
    CONFIDENCE_THRESHOLD_MEDIUM: float = 0.75

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # .env에 모르는 변수가 있어도 ValidationError 내지 않음

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
            logger.info(
                "🔐 AWS environment detected - loading secrets from Secrets Manager"
            )

            # Fetch GEMINI_API_KEY from Secrets Manager
            try:
                gemini_key = get_secret_or_env(
                    secret_name="hairme-gemini-api-key",
                    env_var_name="GEMINI_API_KEY",
                    region_name=self.AWS_REGION,
                    required=True,
                )
                if gemini_key:
                    self.GEMINI_API_KEY = gemini_key
                    logger.info("✅ GEMINI_API_KEY loaded from Secrets Manager")
            except Exception as e:
                logger.error(f"❌ Failed to load GEMINI_API_KEY: {str(e)}")

            # Fetch ADMIN_API_KEY from Secrets Manager
            try:
                admin_key = get_secret_or_env(
                    secret_name="hairme-admin-api-key",
                    env_var_name="ADMIN_API_KEY",
                    region_name=self.AWS_REGION,
                    required=False,
                )
                if admin_key:
                    self.ADMIN_API_KEY = admin_key
                    logger.info("✅ ADMIN_API_KEY loaded from Secrets Manager")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load ADMIN_API_KEY: {str(e)}")

            # Fetch JWT_SECRET_KEY from Secrets Manager
            try:
                jwt_secret = get_secret_or_env(
                    secret_name="hairme-jwt-secret",
                    env_var_name="JWT_SECRET_KEY",
                    region_name=self.AWS_REGION,
                    required=False,
                )
                if jwt_secret:
                    self.JWT_SECRET_KEY = jwt_secret
                    logger.info("✅ JWT_SECRET_KEY loaded from Secrets Manager")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load JWT_SECRET_KEY: {str(e)}")

        else:
            logger.info(
                "💻 Local/Dev environment detected - using environment variables/.env file"
            )

        # Validate required secrets
        if not self.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is required but not found in Secrets Manager or environment variables"
            )


# Singleton instance
settings = Settings()
