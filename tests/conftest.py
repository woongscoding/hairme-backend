"""Pytest configuration and fixtures for testing"""

import os
import sys

# Set environment variables BEFORE importing main
os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_ENABLED", "false")
os.environ.setdefault("ML_MODEL_PATH", "models/test_model.pt")
os.environ["TESTING"] = "true"  # Skip .env file loading during tests

import pytest
import io
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from main import app
from database.models import Base
from config.settings import settings


# ========== Test Database Setup ==========
@pytest.fixture(scope="function")
def test_db():
    """Create a test database session"""
    # Use in-memory SQLite for testing
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create tables
    Base.metadata.create_all(bind=engine)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


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


@pytest.fixture
def mock_ml_model():
    """Mock ML model predictions"""
    with patch("core.ml_loader.predict_ml_score") as mock:
        mock.return_value = (8.5, 0.92)  # (score, confidence)
        yield mock


# Environment variables are set at the top of this file before imports
