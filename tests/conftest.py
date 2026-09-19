"""Pytest configuration and fixtures for testing"""

import os
import sys

# Set environment variables BEFORE importing main
os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("REDIS_ENABLED", "false")
os.environ.setdefault("ML_MODEL_PATH", "models/test_model.pt")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")
os.environ["TESTING"] = "true"  # Skip .env file loading during tests

# settings.USE_DYNAMODB 는 기본값이 False 인데도 Settings.__init__ 의 AWS 환경
# 처리에서 True 로 올라온다. 반면 database.dynamodb_connection.init_dynamodb 는
# os.getenv("USE_DYNAMODB") 를 보므로 False 다. 이 불일치 때문에 /api/health 의
# check_dynamodb 만 실제 DescribeTable 을 호출하고 있었다
# (test_health_check.py 의 latency_ms 플레이크 원인이기도 하다 - 실제 네트워크 지연).
# 테스트에서는 두 경로의 판단을 명시적으로 일치시킨다.
os.environ.setdefault("USE_DYNAMODB", "false")

import pytest
import io
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
from fastapi.testclient import TestClient

from main import app
from config.settings import settings


# ========== 실제 AWS 호출 차단 ==========
# 기본 단위 테스트는 AWS 를 건드리면 안 된다. 목을 빠뜨리면 조용히 운영
# 테이블(hairstyle_usage, hairme-analysis 등)에 붙어버리는데, 코드 대부분이
# 저장소 장애를 fail-open 으로 삼키기 때문에 테스트는 그대로 통과한다.
# (2026-09-19 에 실제로 발생 - docs/OPS_TEST_AWS_INCIDENT.md 참고)
#
# 그래서 botocore 의 API 호출 진입점을 막고, 시도 자체를 즉시 실패로 만든다.
#
# 예외(실제 호출을 허용)는 둘 뿐이다:
#   - RUN_DYNAMODB_INTEGRATION=1  : 의도적인 통합 테스트 (전용 테스트 테이블 필요)
#   - @pytest.mark.aws           : 실제 호출이 목적인 개별 테스트
AWS_INTEGRATION_OPT_IN_ENV = "RUN_DYNAMODB_INTEGRATION"


class RealAWSCallAttempted(BaseException):
    """단위 테스트에서 실제 AWS 호출을 시도했을 때 발생.

    BaseException 을 상속하는 이유: 프로덕션 코드 곳곳의 `except Exception`
    (저장소 장애 시 통과시키는 fail-open 경로)에 삼켜지면 안 되기 때문이다.
    삼켜지면 목을 빠뜨린 테스트가 그대로 통과해 차단 장치가 무의미해진다.
    """


@pytest.fixture(autouse=True)
def block_real_aws_calls(request):
    """모든 테스트에서 실제 AWS API 호출을 차단한다 (위 예외 두 가지 제외)"""
    if os.getenv(AWS_INTEGRATION_OPT_IN_ENV) == "1":
        yield
        return
    if request.node.get_closest_marker("aws"):
        yield
        return

    try:
        import botocore.client
    except ImportError:  # boto3 미설치 환경
        yield
        return

    def _blocked(self, operation_name, api_params):
        service = getattr(getattr(self, "meta", None), "service_model", None)
        service_name = getattr(service, "service_name", "aws")
        raise RealAWSCallAttempted(
            f"단위 테스트가 실제 AWS 호출을 시도했습니다: "
            f"{service_name}.{operation_name}\n"
            f"목을 추가하세요. 실제 호출이 목적이라면 @pytest.mark.aws 를 붙이거나 "
            f"{AWS_INTEGRATION_OPT_IN_ENV}=1 로 실행하세요.\n"
            f"흔한 원인: api/endpoints/* 는 서비스 팩토리를 import 시점에 "
            f"바인딩하므로, core.quota 가 아니라 해당 엔드포인트 모듈의 이름을 "
            f"patch 해야 합니다."
        )

    with patch.object(botocore.client.BaseClient, "_make_api_call", _blocked):
        yield


# ========== Test Client Setup ==========
@pytest.fixture(scope="module")
def client():
    """Create a test client for FastAPI app"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="function")
def client_with_mocks():
    """Create a test client with all external dependencies mocked via FastAPI DI overrides"""
    from core.dependencies import get_face_detection_service, get_hybrid_service

    # Create mock face detection service
    mock_face_detector = Mock()
    mock_mp_features = Mock()
    mock_mp_features.face_shape = "계란형"
    mock_mp_features.skin_tone = "봄웜"
    mock_mp_features.confidence = 0.92
    mock_mp_features.gender = "neutral"
    mock_mp_features.face_features = None
    mock_mp_features.skin_features = None
    mock_face_detector.detect_face.return_value = {
        "has_face": True,
        "face_count": 1,
        "method": "mediapipe",
        "features": mock_mp_features,
    }

    # Create mock ML recommendation service
    mock_ml_recommender = Mock()
    mock_ml_recommender.recommend.return_value = create_mock_recommendations()

    # Override FastAPI dependencies
    app.dependency_overrides[get_face_detection_service] = lambda: mock_face_detector
    app.dependency_overrides[get_hybrid_service] = lambda: mock_ml_recommender

    with patch("core.cache.redis_client") as mock_redis:
        mock_redis.get.return_value = None  # No cache hit

        with TestClient(app) as test_client:
            yield test_client

    # Clean up overrides
    app.dependency_overrides.pop(get_face_detection_service, None)
    app.dependency_overrides.pop(get_hybrid_service, None)


# ========== Mock Data Generators ==========
def create_mock_face_features():
    """Create mock MediaPipe face analysis results"""
    return {
        "face_shape": "계란형",
        "face_shape_confidence": 0.92,
        "face_ratio": 1.45,
        "jawline_angle": 125.3,
        "landmarks": [[0.5, 0.5] for _ in range(478)],
    }


def create_mock_recommendations():
    """Create mock ML recommendation results"""
    return {
        "recommendations": [
            {
                "style_name": "레이어드 컷",
                "reason": "얼굴형과 잘 어울림",
                "confidence": 0.95,
                "ml_score": 8.5,
            },
            {
                "style_name": "시스루 뱅",
                "reason": "이마 비율 보완",
                "confidence": 0.88,
                "ml_score": 8.2,
            },
            {
                "style_name": "웨이브 펌",
                "reason": "부드러운 인상",
                "confidence": 0.82,
                "ml_score": 7.8,
            },
        ],
    }


@pytest.fixture
def sample_image_bytes():
    """Create a sample image for testing"""
    # Create a simple RGB image
    img = Image.new("RGB", (640, 480), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def sample_image_file(sample_image_bytes):
    """Create a sample UploadFile for testing"""
    return {"file": ("test_image.jpg", sample_image_bytes, "image/jpeg")}


# ========== Mock External Services ==========
@pytest.fixture
def mock_gemini_api():
    """Mock Gemini API responses"""
    with patch("google.generativeai.GenerativeModel") as mock:
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = """```json
{
  "face_shape": "계란형",
  "personal_color": "봄웜",
  "recommended_hairstyles": [
    {"name": "레이어드 컷", "reason": "얼굴형과 잘 어울림"},
    {"name": "시스루 뱅", "reason": "이마 비율 보완"},
    {"name": "웨이브 펌", "reason": "부드러운 인상"}
  ]
}
```"""
        mock_model.generate_content.return_value = mock_response
        mock.return_value = mock_model
        yield mock


@pytest.fixture
def mock_mediapipe():
    """Mock MediaPipe face analyzer"""
    with patch("models.mediapipe_analyzer.MediaPipeFaceAnalyzer") as mock:
        mock_analyzer = Mock()
        mock_analyzer.analyze_face.return_value = create_mock_face_features()
        mock.return_value = mock_analyzer
        yield mock


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    with patch("core.cache.redis_client") as mock:
        mock.get.return_value = None
        mock.setex.return_value = True
        mock.ping.return_value = True
        yield mock


# Environment variables are set at the top of this file before imports
