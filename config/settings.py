"""Application settings and configuration management using Pydantic Settings"""

import os
import logging
from typing import Any, Dict, List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from config.secrets import get_secret_or_env, is_aws_environment

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application configuration loaded from environment variables"""

    # API Keys (will be overridden by __init__ if in AWS)
    GEMINI_API_KEY: str = ""
    ADMIN_API_KEY: Optional[str] = None  # For admin endpoints authentication
    OPENAI_API_KEY: Optional[str] = None  # LangGraph 챗봇 (임베딩 + LLM)
    TAVILY_API_KEY: Optional[str] = None  # 챗봇 web_search 노드

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
    # 하루 전체 Gemini 호출 수(api_calls 합계, 캐시 히트 제외)를 기준으로
    # 비로그인 신규 합성을 멈추는 장치. 전체 비용의 하드 상한이 아니다:
    # 호출 후 집계라 동시 요청은 함께 통과하고, 회원은 제한하지 않으며,
    # 집계 저장소 장애 시에도 통과시킨다 (core/synthesis_budget.py 참고).
    # 0(기본)이면 상한 조회를 건너뛴다. 집계는 값과 무관하게 항상 수행한다.
    DAILY_SYNTHESIS_BUDGET: int = 0

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

    # ===== 추천 사유 문구 =====
    # 별점 접두어("★★★ 강력 추천" 등)를 사유 문구에 붙일지 여부.
    # 재학습 모델(v6_feedback_20260909)이 40~65 점대를 내는데 기준은 75/80/85/90
    # 고정이라 모든 카드가 "★☆☆ 추천" 으로 나간다. 점수에 변별력이 생길 때까지
    # 문구에서 빼둔다(기준 재설정은 하지 않는다). True 로 올리면 즉시 되돌아간다.
    SHOW_CONFIDENCE_STARS: bool = False

    # ===== 크레딧 정책 =====
    SIGNUP_BONUS_CREDITS: int = 5  # 가입 보너스 (평생 무료분)
    SYNTHESIS_CREDIT_COST: int = 1  # 합성 1회당 차감 크레딧

    # ===== 합성 출력 등급 (무료 / 유료) =====
    # 유료 = 구매한 크레딧(paid_credits)으로 차감된 합성. 그 밖(비로그인 무료 한도,
    # 가입 보너스, 보상형 광고 크레딧, 관리자 지급)은 전부 무료 등급이다.
    # 값은 Gemini ImageConfig.image_size 로 전달된다 ("512" | "1K" | "2K").
    # 512 를 지원하지 않는 모델(Nano Banana)은 1K 로 받아 finalize 단계에서 512 로 줄인다.
    # 빈 문자열이면 파라미터를 넣지 않는다 (모델 기본값 1K).
    FREE_IMAGE_SIZE: str = "512"
    PAID_IMAGE_SIZE: str = "1K"
    # 무료 등급 결과 우하단에 워터마크를 넣는다. 공유 이미지에 앱 이름이 남는 효과도 있다.
    FREE_RESULT_WATERMARK: bool = True
    RESULT_WATERMARK_TEXT: str = "HairMe"

    # ===== 무료 정책 실험 (수익화 v2.1 Phase 1) =====
    # 실험군(신규 기기의 TREATMENT_PERCENT%): 비회원 평생 N회 + 가입 보너스 축소
    #   + 보상형 광고는 크레딧 대신 원가 0원 기능 해제.
    # 대조군: 기존 정책 그대로 (비회원 매일 DAILY_SYNTHESIS_LIMIT 회, 가입 보너스 5회).
    #
    # 기기 코호트는 처음 본 순간 한 번만 정해지고 바뀌지 않는다
    # (services/free_policy_service.py). START 이전에 처음 본 기기와 START 가
    # 비어 있을 때 본 기기는 "grandfathered" 로 대조군에 고정된다.
    # 따라서 실험 전에 이 코드를 먼저 배포해 두면 기존 사용자가 미리 등록된다.
    #
    # 원복: ENABLED=False 로 내리면 기록된 코호트와 무관하게 전원 대조군 정책으로 돌아간다.
    FREE_POLICY_EXPERIMENT_ENABLED: bool = False
    FREE_POLICY_EXPERIMENT_START: str = ""  # ISO8601 (예: 2026-11-20T00:00:00+09:00)
    FREE_POLICY_TREATMENT_PERCENT: int = 50
    FREE_POLICY_SALT: str = "free-policy-2026-11"
    TREATMENT_GUEST_LIFETIME_LIMIT: int = 1
    TREATMENT_SIGNUP_BONUS_CREDITS: int = 3
    # analyze-hybrid data.full_ranking 길이 (보상형 광고 해제 기능 "전체 스타일 랭킹").
    # 0 이면 랭킹을 계산하지 않는다 (빈 배열).
    FULL_RANKING_LIMIT: int = 80

    # ===== 추천 Top-3 동점권 탐색 (services/rec_exploration.py) =====
    # 켜면 analyze-hybrid 가 Top-POOL 을 받아 top - MARGIN(원점수) 이상인 동점권에서
    # softmax(score / TEMPERATURE) 로 3개를 뽑는다. 사용자(없으면 image_hash) 해시
    # 버킷이 RATIO 미만이면 실험군, 나머지는 대조군(결정적 Top-3, 노출 메타만 기록).
    # 끄면 기존 k=3 호출 그대로. full_ranking 은 어느 쪽이든 결정적 순서.
    REC_EXPLORE_ENABLED: bool = False
    REC_EXPLORE_RATIO: float = 0.5
    REC_EXPLORE_POOL: int = 20
    REC_EXPLORE_MARGIN: float = 3.0
    REC_EXPLORE_TEMPERATURE: float = 1.5

    # ===== 추천 반응률 보너스 (services/rec_engagement.py) =====
    # 켜면 s3://{MLOPS_S3_BUCKET}/{REC_ENGAGEMENT_S3_KEY} (scripts/build_style_engagement.py
    # 가 매일 만든 "성별 x 얼굴형 그룹의 스타일 반응률")를 읽어, 실험군(사용자/image_hash
    # 해시 버킷 < RATIO, 탐색 실험과 독립 배정)의 원점수에
    #   clip(BETA * (셀 반응률 - 그룹 평균) / 그룹 평균, -CAP, +CAP)  점수 포인트를 더한다.
    # 보너스는 탐색(동점권) 선택 전에 더해져 순위·동점권 모두에 영향을 준다.
    # 꺼짐 / 대조군 / 테이블 없음은 응답이 기존과 같다 (새 필드 없음).
    REC_ENGAGEMENT_ENABLED: bool = False
    REC_ENGAGEMENT_RATIO: float = 0.5
    REC_ENGAGEMENT_BETA: float = 3.0  # lift +1 (반응 2배) -> +3점 (CAP 에서 잘린다)
    # 보너스 절대 상한(점). 두 스타일의 상대 순서는 최대 2*CAP 까지 움직이므로
    # CAP = REC_EXPLORE_MARGIN / 2 이면 보너스만으로는 원점수 차이가 동점권 폭(3점)
    # 이내인 스타일끼리만 순서가 바뀐다. 근거는 scripts/build_style_engagement.py 머리말
    REC_ENGAGEMENT_CAP: float = 1.5
    REC_ENGAGEMENT_S3_KEY: str = "engagement/current.json"
    REC_ENGAGEMENT_CACHE_TTL_SECONDS: int = 3600
    # generated_at 이 이 일수보다 오래된 테이블은 쓰지 않는다 (매일 집계가 멈춘 경우)
    REC_ENGAGEMENT_MAX_AGE_DAYS: float = 3.0
    # 실험군 사유 문구 교체 조건: 셀 lift(그룹 평균 기준) >= MIN_LIFT, 셀 노출 >=
    # MIN_EXPOSURES, save/share 한 **로그인 사용자** 수 >= MIN_SUBJECTS ("많이 저장한" 의
    # 근거. 합성·좋아요만으로는 나가지 않고, 비로그인은 사진만 바꾸면 새 주체가 되어
    # 위조가 쉬우므로 세지 않는다)
    REC_ENGAGEMENT_REASON_MIN_LIFT: float = 0.3
    REC_ENGAGEMENT_REASON_MIN_EXPOSURES: int = 50
    REC_ENGAGEMENT_REASON_MIN_SUBJECTS: int = 15

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
    # credits_100 은 공헌이익이 거의 남지 않아 판매하지 않는다 (수익화 v2.1).
    # starter_5 는 계정당 1회 (api/endpoints/credits.py ONE_PER_ACCOUNT_PRODUCTS).
    CREDIT_PRODUCTS: Dict[str, int] = {
        "starter_5": 5,
        "credits_10": 10,
        "credits_30": 30,
    }
    # credits_30 노출 A/B: 이 비율(%)의 회원에게 "추천" 배지를 붙인다 (나머지는 보조 노출)
    PAYWALL_RECOMMENDED_BADGE_PERCENT: int = 50

    # ===== Google Play 환불/취소 회수 (voidedpurchases 폴링) =====
    PLAY_VOID_RECLAIM_ENABLED: bool = (
        True  # 잡 이벤트 {"job":"reclaim_voided_purchases"} 처리 여부
    )
    PLAY_VOID_LOOKBACK_DAYS: int = 30  # 폴링 시 조회할 과거 기간 (Play 최대 30일)

    # ===== AdMob 리워드 광고 (SSV) =====
    # 유저당 하루 보상 횟수 상한.
    #
    # 5 -> 3 하향: 보상 1회 = 크레딧 1개 = 합성 1회인데, 한국 eCPM $3~8 기준
    # 보상형 노출 수익은 7원 안팎이고 합성 1회 Gemini 비용은 60~95원이다
    # (gemini-3.1-flash-image, 출력 1장). 즉 보상형 노출은 구조적으로 10배 적자이고,
    # 이 값은 "한 유저가 하루에 만들 수 있는 적자"의 상한으로 기능한다.
    # 광고 수익을 늘리는 수단이 아니라, 크레딧이 소진된 회원을 앱에 남기는
    # 밸브로만 쓴다.
    REWARD_AD_DAILY_LIMIT: int = 3
    # True: 보상 1회 = 합성 크레딧 1개 (대조군). False: 크레딧 대신 원가 0원 기능
    # (퍼스널컬러 상세·전체 랭킹)을 당일 자정까지 해제. 실험군은 이 값과 무관하게
    # 항상 기능 해제다. 무료 정책 실험이 끝나면 False 로 내려 전체를 전환한다.
    REWARD_AD_GRANTS_CREDIT: bool = True
    # 우리 앱의 AdMob 리워드 광고 단위 ID 허용목록 (쉼표 구분).
    # SSV 검증 키는 전 퍼블리셔 공용이므로, 비어 있으면 프로덕션에서 콜백을 거부한다.
    ADMOB_REWARD_AD_UNIT_IDS: str = ""

    # ===== 제휴 커머스 (쿠팡파트너스) =====
    # 클릭 로그 테이블 (PK: user_id, SK: sk) - 콘솔에서 생성 필요
    DYNAMODB_AFFILIATE_CLICKS_TABLE_NAME: str = "hairme-affiliate-clicks"

    # MLOps Configuration
    # 재학습 임계치/Trainer Lambda/SNS 토픽은 services/mlops 가 os.getenv 로 직접 읽는다.
    MLOPS_ENABLED: bool = False  # MLOps 파이프라인 활성화
    MLOPS_S3_BUCKET: str = "hairme-mlops"  # MLOps S3 버킷 (rec_engagement 가 읽음)

    # Security Settings
    ALLOWED_ORIGINS: str = "http://localhost:3000"  # Comma-separated list

    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse ALLOWED_ORIGINS string into list"""
        return [
            origin.strip()
            for origin in self.ALLOWED_ORIGINS.split(",")
            if origin.strip()
        ]

    # Gemini Model Configuration
    # 이미지 생성(합성/염색) 전용 모델.
    # gemini-3.1-flash-image 는 2026-10-06 지원 종료 예고(종료일 미정) → Nano Banana 2.1.
    # 11/06 비교(운영 프롬프트, 6장): 얼굴·구도 유지 개선, 장당 약 $0.067 → $0.036,
    # 생각 minimal 기준 10~12초(이전 9~11초). 롤백은 이 값만 gemini-3.1-flash-image 로.
    GEMINI_IMAGE_MODEL: str = "gemini-nano-banana-2.1"
    # Nano Banana 생각 수준 ("minimal" | "medium" ... , 빈 값이면 모델 기본값 medium).
    # 기본 medium 은 최대 27초까지 걸려 minimal 로 고정한다. low 는 이 모델이 거부한다.
    GEMINI_IMAGE_THINKING_LEVEL: str = "minimal"

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

    # .env에 모르는 변수가 있어도 ValidationError 내지 않음
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

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
