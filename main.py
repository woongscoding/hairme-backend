import os
import time
import json
import logging
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from enum import Enum
import google.generativeai as genai
from PIL import Image
import io
import cv2
import numpy as np
import hashlib
import redis
import urllib.parse
import torch
import torch.nn as nn
import pickle

# ========== SQLAlchemy 추가 ==========
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, create_engine, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ========== 얼굴 분석 모듈 임포트 ==========
# from models.face_analyzer import extract_face_features, create_enhanced_prompt, FaceFeatures  # ❌ 제거됨 (Haar Cascade)
from models.mediapipe_analyzer import MediaPipeFaceAnalyzer, MediaPipeFaceFeatures

# ========== ML & 하이브리드 추천 시스템 임포트 ==========
from services.hybrid_recommender import get_hybrid_service
from models.ml_recommender import get_ml_recommender
from services.feedback_collector import get_feedback_collector
from services.retrain_queue import get_retrain_queue

# ========== 라우터 임포트 ==========
from routers.admin import router as admin_router

Base = declarative_base()

# ========== 로깅 설정 ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== ML 모델 정의 ==========
class HairstyleRecommender(nn.Module):
    """헤어스타일 추천 ML 모델"""

    def __init__(self, n_faces=5, n_skins=3, n_styles=6,
                 emb_dim=16, hidden_dim=64):
        super().__init__()

        self.face_emb = nn.Embedding(n_faces, emb_dim)
        self.skin_emb = nn.Embedding(n_skins, emb_dim)
        self.style_emb = nn.Embedding(n_styles, emb_dim)

        self.shared_layers = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.feedback_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, face, skin, style):
        face_emb = self.face_emb(face)
        skin_emb = self.skin_emb(skin)
        style_emb = self.style_emb(style)

        x = torch.cat([face_emb, skin_emb, style_emb], dim=1)
        shared = self.shared_layers(x)

        score_pred = self.score_head(shared).squeeze(-1)
        feedback_logits = self.feedback_head(shared)

        return score_pred, feedback_logits


# ========== 전역 변수 (모델 & 인코더) ==========
ml_model = None
face_encoder = None
skin_encoder = None
style_encoder = None
sentence_transformer = None
mediapipe_analyzer = None  # MediaPipe 얼굴 분석기
hybrid_service = None  # 하이브리드 추천 서비스 (Gemini + ML)
feedback_collector = None  # 피드백 수집기
retrain_queue = None  # 재학습 큐


# ========== CloudWatch Logs 구조화 로깅 ==========
def log_structured(event_type: str, data: dict):
    """CloudWatch Logs Insights로 분석 가능한 JSON 로그 생성"""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        **data
    }
    logger.info(json.dumps(log_entry, ensure_ascii=False))


def calculate_image_hash(image_data: bytes) -> str:
    """이미지의 SHA256 해시 생성 (캐싱 키로 사용)"""
    return hashlib.sha256(image_data).hexdigest()


# ========== 모델 로더 ==========
def load_ml_model():
    """ML 모델 및 인코더 로드"""
    global ml_model, face_encoder, skin_encoder, style_encoder

    try:
        model_path = 'models/final_model.pth'
        encoder_path = 'models/encoders.pkl'

        # 파일 존재 확인
        if not os.path.exists(model_path):
            logger.warning(f"⚠️ 모델 파일 없음: {model_path}")
            return False

        if not os.path.exists(encoder_path):
            logger.warning(f"⚠️ 인코더 파일 없음: {encoder_path}")
            return False

        # 모델 로드
        ml_model = HairstyleRecommender()
        ml_model.load_state_dict(torch.load(
            model_path,
            map_location=torch.device('cpu')
        ))
        ml_model.eval()

        # 인코더 로드
        with open(encoder_path, 'rb') as f:
            encoders = pickle.load(f)
            face_encoder = encoders['face']
            skin_encoder = encoders['skin']
            style_encoder = encoders['style']

        logger.info("✅ ML 모델 로드 성공")
        logger.info(f"  - 얼굴형: {len(face_encoder.classes_)}개")
        logger.info(f"  - 피부톤: {len(skin_encoder.classes_)}개")
        logger.info(f"  - 스타일: {len(style_encoder.classes_)}개")

        return True

    except Exception as e:
        logger.error(f"❌ ML 모델 로드 실패: {str(e)}")
        ml_model = None
        return False


def load_sentence_transformer():
    """Sentence Transformer 모델 로드 (헤어스타일 임베딩용)"""
    global sentence_transformer

    try:
        from sentence_transformers import SentenceTransformer

        model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        logger.info(f"⏳ Sentence Transformer 로드 중: {model_name}")

        sentence_transformer = SentenceTransformer(model_name)

        logger.info("✅ Sentence Transformer 로드 성공")
        logger.info(f"  - 모델: {model_name}")
        logger.info(f"  - 임베딩 차원: 384")

        return True

    except ImportError:
        logger.warning("⚠️ sentence-transformers 라이브러리가 설치되지 않음")
        logger.warning("  - pip install sentence-transformers 실행 필요")
        sentence_transformer = None
        return False

    except Exception as e:
        logger.error(f"❌ Sentence Transformer 로드 실패: {str(e)}")
        sentence_transformer = None
        return False


# ========== 예측 함수 ==========
def predict_ml_score(face_shape: str, skin_tone: str, hairstyle: str) -> float:
    """
    ML 모델로 점수 예측

    Args:
        face_shape: 얼굴형 (예: "계란형")
        skin_tone: 피부톤 (예: "봄웜")
        hairstyle: 헤어스타일 (예: "시스루뱅 단발")

    Returns:
        예측 점수 (0.0 ~ 1.0)
    """
    if ml_model is None:
        return 0.85  # 모델 없으면 기본값

    try:
        # ========== Gemini 출력 → ML 입력 매핑 ==========
        skin_tone_mapping = {
            "봄웜": "웜톤",
            "가을웜": "웜톤",
            "여름쿨": "쿨톤",
            "겨울쿨": "쿨톤"
        }

        mapped_skin = skin_tone_mapping.get(skin_tone, "중간톤")

        # ========== 인코딩 ==========
        try:
            face_encoded = face_encoder.transform([face_shape])[0]
        except ValueError:
            logger.warning(f"⚠️ 알 수 없는 얼굴형: {face_shape}, 기본값 사용")
            face_encoded = 1  # 계란형

        try:
            skin_encoded = skin_encoder.transform([mapped_skin])[0]
        except ValueError:
            logger.warning(f"⚠️ 알 수 없는 피부톤: {mapped_skin}, 기본값 사용")
            skin_encoded = 1  # 중간톤

        try:
            style_encoded = style_encoder.transform([hairstyle])[0]
        except ValueError:
            logger.warning(f"⚠️ 알 수 없는 스타일: {hairstyle}, 기본값 사용")
            style_encoded = 2  # 시스루뱅 단발

        # ========== Tensor 변환 ==========
        face_tensor = torch.tensor([face_encoded], dtype=torch.long)
        skin_tensor = torch.tensor([skin_encoded], dtype=torch.long)
        style_tensor = torch.tensor([style_encoded], dtype=torch.long)

        # ========== 예측 ==========
        with torch.no_grad():
            score_pred, _ = ml_model(face_tensor, skin_tensor, style_tensor)
            score = score_pred.item()

        logger.info(f"ML 예측: {face_shape} + {mapped_skin} + {hairstyle} → {score:.3f}")
        return round(score, 3)

    except Exception as e:
        logger.error(f"ML 예측 실패: {str(e)}")
        return 0.85


def get_confidence_level(score: float) -> str:
    """점수를 신뢰도 레벨로 변환"""
    if score >= 0.90:
        return "매우 높음"
    elif score >= 0.85:
        return "높음"
    elif score >= 0.75:
        return "보통"
    else:
        return "낮음"


# ========== 피드백 Enum ==========
class FeedbackType(str, Enum):
    LIKE = "like"
    DISLIKE = "dislike"


# ========== 데이터베이스 모델 ==========
class AnalysisHistory(Base):
    """분석 기록 테이블 - v20.1 (ML 통합)"""
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), default="anonymous")
    image_hash = Column(String(64), index=True)
    face_shape = Column(String(50))
    personal_color = Column(String(50))
    recommendations = Column(JSON)
    processing_time = Column(Float)
    detection_method = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # OpenCV 측정 데이터 - 수평 비율
    opencv_face_ratio = Column(Float)
    opencv_forehead_ratio = Column(Float)
    opencv_cheekbone_ratio = Column(Float)
    opencv_jaw_ratio = Column(Float)
    opencv_prediction = Column(String(50))
    opencv_confidence = Column(Float)
    opencv_gemini_agreement = Column(Boolean)

    # OpenCV 측정 데이터 - 수직 비율 (v20.1.6)
    opencv_upper_face_ratio = Column(Float)
    opencv_middle_face_ratio = Column(Float)
    opencv_lower_face_ratio = Column(Float)

    # v20: 추천 스타일 저장
    recommended_styles = Column(JSON)

    # v20: 피드백 컬럼 (String으로 저장하여 타입 불일치 방지)
    style_1_feedback = Column(String(10), nullable=True)
    style_2_feedback = Column(String(10), nullable=True)
    style_3_feedback = Column(String(10), nullable=True)

    # v20: 네이버 클릭 여부
    style_1_naver_clicked = Column(Boolean, default=False)
    style_2_naver_clicked = Column(Boolean, default=False)
    style_3_naver_clicked = Column(Boolean, default=False)

    # v20: 피드백 제출 시각
    feedback_at = Column(DateTime, nullable=True)


# ========== Pydantic 모델 ==========
class FeedbackRequest(BaseModel):
    """피드백 제출 요청"""
    analysis_id: int = Field(..., description="분석 결과 ID")
    style_index: int = Field(..., ge=1, le=3, description="스타일 인덱스 (1, 2, 3)")
    feedback: FeedbackType = Field(..., description="좋아요 또는 싫어요")
    naver_clicked: bool = Field(default=False, description="네이버 이미지 검색 클릭 여부")


class FeedbackResponse(BaseModel):
    """피드백 제출 응답"""
    success: bool
    message: str
    analysis_id: int
    style_index: int


# ========== FastAPI 앱 초기화 ==========
app = FastAPI(
    title="HairMe API",
    description="AI 기반 헤어스타일 추천 서비스 (v20.1.6: 수직 비율 데이터 수집)",
    version="20.1.6"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 라우터 등록 ==========
app.include_router(admin_router, prefix="/api", tags=["admin"])


# ========== 앱 시작 이벤트 ==========
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 ML 모델 로드"""
    global mediapipe_analyzer, hybrid_service, feedback_collector, retrain_queue

    logger.info("🚀 서버 시작 중...")

    # MediaPipe 얼굴 분석기 초기화
    try:
        mediapipe_analyzer = MediaPipeFaceAnalyzer()
        logger.info("✅ MediaPipe 얼굴 분석기 초기화 완료")
        log_structured("mediapipe_initialized", {
            "status": "success",
            "landmarks": 478
        })
    except Exception as e:
        logger.error(f"❌ MediaPipe 초기화 실패: {str(e)}")
        mediapipe_analyzer = None
        log_structured("mediapipe_initialized", {
            "status": "failed",
            "error": str(e)
        })

    # ML 모델 로드 시도
    ml_loaded = load_ml_model()

    if ml_loaded:
        logger.info("✅ ML 모드: 활성화")
        log_structured("ml_model_loaded", {
            "status": "success",
            "model_path": "models/final_model.pth"
        })
    else:
        logger.warning("⚠️ ML 모드: 비활성화 (기본 점수 사용)")
        log_structured("ml_model_loaded", {
            "status": "failed",
            "fallback": "default_score"
        })

    # Sentence Transformer 로드 시도
    st_loaded = load_sentence_transformer()

    if st_loaded:
        logger.info("✅ 스타일 임베딩: 활성화")
        log_structured("sentence_transformer_loaded", {
            "status": "success",
            "model": "paraphrase-multilingual-MiniLM-L12-v2",
            "embedding_dim": 384
        })
    else:
        logger.warning("⚠️ 스타일 임베딩: 비활성화 (임베딩 없이 진행)")
        log_structured("sentence_transformer_loaded", {
            "status": "failed",
            "fallback": "no_embedding"
        })

    # 하이브리드 추천 서비스 초기화 (Gemini + ML)
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            hybrid_service = get_hybrid_service(gemini_api_key)
            logger.info("✅ 하이브리드 추천 서비스 초기화 완료 (Gemini + ML)")
            log_structured("hybrid_service_initialized", {
                "status": "success",
                "gemini_model": "gemini-1.5-flash-latest",
                "ml_model": "hairstyle_recommender.pt"
            })
        else:
            logger.warning("⚠️ GEMINI_API_KEY 없음 - 하이브리드 서비스 비활성화")
            hybrid_service = None
    except Exception as e:
        logger.error(f"❌ 하이브리드 서비스 초기화 실패: {str(e)}")
        hybrid_service = None
        log_structured("hybrid_service_initialized", {
            "status": "failed",
            "error": str(e)
        })

    # 피드백 수집기 초기화
    try:
        feedback_collector = get_feedback_collector()
        logger.info("✅ 피드백 수집기 초기화 완료")
        log_structured("feedback_collector_initialized", {
            "status": "success"
        })
    except Exception as e:
        logger.error(f"❌ 피드백 수집기 초기화 실패: {str(e)}")
        feedback_collector = None
        log_structured("feedback_collector_initialized", {
            "status": "failed",
            "error": str(e)
        })

    # 재학습 큐 초기화
    try:
        retrain_queue = get_retrain_queue()
        logger.info("✅ 재학습 큐 초기화 완료")
        log_structured("retrain_queue_initialized", {
            "status": "success"
        })
    except Exception as e:
        logger.error(f"❌ 재학습 큐 초기화 실패: {str(e)}")
        retrain_queue = None
        log_structured("retrain_queue_initialized", {
            "status": "failed",
            "error": str(e)
        })


# ========== 데이터베이스 연결 ==========
SessionLocal = None
DATABASE_URL = os.getenv("DATABASE_URL")
DB_PASSWORD = os.getenv("DB_PASSWORD")


def migrate_database_schema():
    """
    v20 스키마 마이그레이션 (자동 실행)
    - 필요한 컬럼이 없으면 자동으로 추가
    - 이미 존재하면 스킵
    """
    if not SessionLocal:
        return

    try:
        from sqlalchemy import text
        db = SessionLocal()

        logger.info("🔄 DB 스키마 마이그레이션 시작...")

        # 현재 테이블 구조 확인
        result = db.execute(text("DESCRIBE analysis_history"))
        existing_columns = {row[0] for row in result}

        required_columns = [
            "recommended_styles",
            "style_1_feedback",
            "style_2_feedback",
            "style_3_feedback",
            "style_1_naver_clicked",
            "style_2_naver_clicked",
            "style_3_naver_clicked",
            "feedback_at",
            # v20.1.6: 수직 비율 데이터
            "opencv_upper_face_ratio",
            "opencv_middle_face_ratio",
            "opencv_lower_face_ratio"
        ]

        missing_columns = [col for col in required_columns if col not in existing_columns]

        if not missing_columns:
            logger.info("✅ 스키마가 이미 최신 상태입니다 (v20.1.6)")
            db.close()
            return

        logger.info(f"🔧 누락된 컬럼 발견: {missing_columns}")

        # 마이그레이션 SQL 실행
        migration_sqls = []

        if "recommended_styles" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN recommended_styles JSON COMMENT '추천된 3개 헤어스타일'"
            )

        if "style_1_feedback" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN style_1_feedback ENUM('like', 'dislike') DEFAULT NULL"
            )

        if "style_2_feedback" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN style_2_feedback ENUM('like', 'dislike') DEFAULT NULL"
            )

        if "style_3_feedback" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN style_3_feedback ENUM('like', 'dislike') DEFAULT NULL"
            )

        if "style_1_naver_clicked" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN style_1_naver_clicked BOOLEAN DEFAULT FALSE"
            )

        if "style_2_naver_clicked" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN style_2_naver_clicked BOOLEAN DEFAULT FALSE"
            )

        if "style_3_naver_clicked" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN style_3_naver_clicked BOOLEAN DEFAULT FALSE"
            )

        if "feedback_at" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN feedback_at DATETIME DEFAULT NULL"
            )

        # v20.1.6: 수직 비율 컬럼 추가
        if "opencv_upper_face_ratio" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN opencv_upper_face_ratio FLOAT DEFAULT NULL COMMENT '상안부 높이 비율'"
            )

        if "opencv_middle_face_ratio" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN opencv_middle_face_ratio FLOAT DEFAULT NULL COMMENT '중안부 높이 비율'"
            )

        if "opencv_lower_face_ratio" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN opencv_lower_face_ratio FLOAT DEFAULT NULL COMMENT '하안부 높이 비율'"
            )

        # 트랜잭션으로 실행
        for sql in migration_sqls:
            logger.info(f"실행: {sql[:80]}...")
            db.execute(text(sql))

        db.commit()
        logger.info("✅ 스키마 마이그레이션 완료!")

        log_structured("schema_migration", {
            "status": "success",
            "added_columns": missing_columns
        })

        db.close()

    except Exception as e:
        logger.error(f"❌ 스키마 마이그레이션 실패: {str(e)}")
        logger.error("서버는 계속 실행되지만, v20 기능이 제대로 작동하지 않을 수 있습니다.")
        if 'db' in locals():
            db.rollback()
            db.close()


if DATABASE_URL and DB_PASSWORD:
    try:
        sync_db_url = DATABASE_URL.replace("asyncmy", "pymysql").replace("://admin@", f"://admin:{DB_PASSWORD}@")
        engine = create_engine(
            sync_db_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
        SessionLocal = sessionmaker(bind=engine)
        Base.metadata.create_all(bind=engine)
        logger.info("✅ MySQL 데이터베이스 연결 성공")

        # 🆕 자동 스키마 마이그레이션 실행
        migrate_database_schema()

        log_structured("database_connected", {
            "database": "hairme-data",
            "tables": ["analysis_history"]
        })
    except Exception as e:
        logger.error(f"❌ MySQL 연결 실패: {str(e)}")
        SessionLocal = None
else:
    logger.warning("⚠️ DATABASE_URL 또는 DB_PASSWORD가 설정되지 않았습니다.")

# ========== Redis 캐시 ==========
redis_client = None
REDIS_URL = os.getenv("REDIS_URL")
CACHE_TTL = 86400

if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info(f"✅ Redis 연결 성공: {REDIS_URL}")
    except Exception as e:
        logger.error(f"❌ Redis 연결 실패: {str(e)}")
        redis_client = None
else:
    logger.warning("⚠️ REDIS_URL 환경변수가 설정되지 않았습니다.")


def get_cached_result(image_hash: str) -> Optional[dict]:
    """Redis에서 캐시된 분석 결과 조회"""
    if not redis_client:
        return None
    try:
        cached = redis_client.get(f"analysis:{image_hash}")
        if cached:
            log_structured("cache_hit", {"image_hash": image_hash[:16]})
            return json.loads(cached)
        return None
    except Exception as e:
        logger.error(f"Redis 조회 중 오류: {str(e)}")
        return None


def save_to_cache(image_hash: str, result: dict):
    """Redis에 분석 결과 저장"""
    if not redis_client:
        return
    try:
        redis_client.setex(
            f"analysis:{image_hash}",
            CACHE_TTL,
            json.dumps(result, ensure_ascii=False)
        )
    except Exception as e:
        logger.error(f"Redis 저장 중 오류: {str(e)}")


# ========== Gemini API 초기화 ==========
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY 환경변수가 설정되지 않았습니다!")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("✅ Gemini API 초기화 완료")
    except Exception as e:
        logger.error(f"Gemini API 초기화 실패: {str(e)}")

MODEL_NAME = "gemini-flash-latest"

# ========== Haar Cascade 제거됨 (MediaPipe로 대체) ==========
# MediaPipe가 실패하면 Gemini로 직접 백업

# ========== Gemini 기본 프롬프트 ==========
ANALYSIS_PROMPT = """분석하고 JSON으로 응답:

얼굴형: 계란형/둥근형/각진형/긴형 중 1개
퍼스널컬러: 봄웜/가을웜/여름쿨/겨울쿨 중 1개
헤어스타일 추천 3개 (각 이름 15자, 이유 30자 이내)

JSON 형식:
{
  "analysis": {
    "face_shape": "계란형",
    "personal_color": "봄웜",
    "features": "이목구비 특징"
  },
  "recommendations": [
    {"style_name": "스타일명", "reason": "추천 이유"}
  ]
}"""


# ========== 헬퍼 함수들 ==========
def verify_face_with_gemini(image_data: bytes) -> dict:
    """OpenCV 실패 시 Gemini로 빠른 얼굴 검증"""
    try:
        image = Image.open(io.BytesIO(image_data))
        image.thumbnail((256, 256))

        model = genai.GenerativeModel(MODEL_NAME)
        prompt = """이미지에 사람 얼굴이 있나요?

JSON으로만 답변:
{"has_face": true/false, "face_count": 숫자}"""

        response = model.generate_content([prompt, image])
        result = json.loads(response.text.strip())

        return {
            "has_face": result.get("has_face", False),
            "face_count": result.get("face_count", 0),
            "method": "gemini"
        }

    except Exception as e:
        logger.error(f"Gemini 얼굴 검증 실패: {str(e)}")
        return {
            "has_face": False,
            "face_count": 0,
            "method": "gemini",
            "error": str(e)
        }


def detect_face(image_data: bytes) -> dict:
    """얼굴 감지 (MediaPipe 우선, 실패 시 Gemini)"""
    # 1차 시도: MediaPipe (가장 정확함 - 90%+)
    if mediapipe_analyzer is not None:
        try:
            mp_features = mediapipe_analyzer.analyze(image_data)

            if mp_features:
                log_structured("face_detection", {
                    "method": "mediapipe",
                    "face_count": 1,
                    "success": True,
                    "face_shape": mp_features.face_shape,
                    "skin_tone": mp_features.skin_tone,
                    "confidence": mp_features.confidence
                })
                return {
                    "has_face": True,
                    "face_count": 1,
                    "method": "mediapipe",
                    "features": mp_features  # MediaPipe 분석 결과 포함
                }

        except Exception as e:
            logger.warning(f"MediaPipe 얼굴 감지 실패: {str(e)}")

    # 2차 시도: Gemini (최종 백업)
    logger.info("MediaPipe 실패, Gemini로 얼굴 검증 시작...")
    gemini_result = verify_face_with_gemini(image_data)

    log_structured("face_detection", {
        "method": "gemini",
        "face_count": gemini_result.get("face_count", 0),
        "success": gemini_result.get("has_face", False)
    })

    return gemini_result


def analyze_with_gemini(image_data: bytes, mp_features: Optional[MediaPipeFaceFeatures] = None) -> dict:
    """Gemini Vision API로 얼굴 분석 (MediaPipe 힌트 제공)"""
    try:
        image = Image.open(io.BytesIO(image_data))

        # MediaPipe 결과가 있으면 힌트 제공
        if mp_features:
            prompt = f"""다음 얼굴 사진을 분석하고 JSON으로 응답해주세요.

🔍 **참고용 측정 데이터** (MediaPipe AI 분석 - 신뢰도 {mp_features.confidence:.0%}):
- 얼굴형: {mp_features.face_shape}
- 피부톤: {mp_features.skin_tone}
- 얼굴 비율(높이/너비): {mp_features.face_ratio:.2f}
- 이마 너비: {mp_features.forehead_width:.0f}px
- 광대 너비: {mp_features.cheekbone_width:.0f}px
- 턱 너비: {mp_features.jaw_width:.0f}px
- ITA 값: {mp_features.ITA_value:.1f}°

위 수치는 참고만 하고, 당신의 시각적 판단을 우선하세요.

**분석 항목:**
1. 얼굴형: 계란형/둥근형/각진형/긴형/하트형 중 1개
2. 퍼스널컬러: 봄웜/가을웜/여름쿨/겨울쿨 중 1개
3. 헤어스타일 추천 3개 (각 이름 15자, 이유 30자 이내)

**JSON 형식:**
{{
  "analysis": {{
    "face_shape": "계란형",
    "personal_color": "봄웜",
    "features": "이목구비 특징 설명"
  }},
  "recommendations": [
    {{"style_name": "스타일명", "reason": "추천 이유"}}
  ]
}}"""
            logger.info(f"✅ MediaPipe 힌트 적용: {mp_features.face_shape} / {mp_features.skin_tone}")

        else:
            # MediaPipe 없을 때 기본 프롬프트
            prompt = ANALYSIS_PROMPT
            logger.warning("⚠️ MediaPipe 특징 없음, 기본 프롬프트 사용")

        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content([prompt, image])

        raw_text = response.text.strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

        result = json.loads(raw_text.strip())

        logger.info(f"✅ Gemini 분석 성공: {result.get('analysis', {}).get('face_shape')}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 실패: {str(e)}\n응답 내용: {response.text[:200]}")
        raise HTTPException(
            status_code=500,
            detail=f"AI 응답 파싱 실패: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Gemini 분석 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"AI 분석 중 오류가 발생했습니다: {str(e)}"
        )


def save_to_database(
        image_hash: str,
        analysis_result: dict,
        processing_time: float,
        detection_method: str,
        mp_features: Optional[MediaPipeFaceFeatures] = None
) -> Optional[int]:
    """분석 결과를 MySQL에 저장하고 ID 반환"""
    if not SessionLocal:
        logger.warning("⚠️ 데이터베이스 연결이 없어 저장을 생략합니다.")
        return None

    try:
        db = SessionLocal()

        gemini_shape = analysis_result.get("analysis", {}).get("face_shape")

        # MediaPipe 일치도 계산
        mediapipe_agreement = None
        if mp_features:
            mediapipe_agreement = (
                    mp_features.face_shape in gemini_shape or
                    gemini_shape in mp_features.face_shape
            )

        recommendations = analysis_result.get("recommendations", [])

        history = AnalysisHistory(
            image_hash=image_hash,
            face_shape=gemini_shape,
            personal_color=analysis_result.get("analysis", {}).get("personal_color"),
            recommendations=recommendations,
            recommended_styles=recommendations,
            processing_time=processing_time,
            detection_method=detection_method,
            # OpenCV 데이터는 더이상 수집하지 않음 (MediaPipe로 대체)
            opencv_face_ratio=None,
            opencv_forehead_ratio=None,
            opencv_cheekbone_ratio=None,
            opencv_jaw_ratio=None,
            opencv_prediction=None,
            opencv_confidence=None,
            opencv_gemini_agreement=mediapipe_agreement,  # MediaPipe 일치도로 대체
            opencv_upper_face_ratio=None,
            opencv_middle_face_ratio=None,
            opencv_lower_face_ratio=None
        )

        db.add(history)
        db.commit()
        db.refresh(history)

        logger.info(f"✅ DB 저장 성공 (ID: {history.id})")
        log_structured("database_saved", {
            "record_id": history.id,
            "mediapipe_enabled": mp_features is not None,
            "mediapipe_agreement": mediapipe_agreement,
            "recommendations_count": len(recommendations)
        })

        db.close()
        return history.id

    except Exception as e:
        logger.error(f"❌ DB 저장 실패: {str(e)}")
        return None  # ✅ 에러 시 None 반환


# ========== API 엔드포인트 ==========
@app.get("/")
async def root():
    """Root 엔드포인트"""
    mediapipe_status = "enabled" if mediapipe_analyzer is not None else "disabled"
    return {
        "message": "헤어스타일 분석 API - v20.2.0 (MediaPipe 전환 완료)",
        "version": "20.2.0",
        "model": MODEL_NAME,
        "status": "running",
        "features": {
            "mediapipe_analysis": mediapipe_status,
            "gemini_analysis": "enabled" if GEMINI_API_KEY else "disabled",
            "redis_cache": "enabled" if redis_client else "disabled",
            "database": "enabled" if SessionLocal else "disabled",
            "feedback_system": "enabled",
            "ml_prediction": "enabled" if ml_model else "disabled",
            "style_embedding": "enabled" if sentence_transformer else "disabled"
        }
    }


@app.get("/api/health")
async def health_check():
    """헬스체크 엔드포인트"""
    mediapipe_status = "enabled" if mediapipe_analyzer is not None else "disabled"

    return {
        "status": "healthy",
        "version": "20.2.0",
        "model": MODEL_NAME,
        "mediapipe_analysis": mediapipe_status,
        "gemini_api": "configured" if GEMINI_API_KEY else "not_configured",
        "redis": "connected" if redis_client else "disconnected",
        "database": "connected" if SessionLocal else "disconnected",
        "feedback_system": "enabled",
        "ml_model": "enabled" if ml_model else "disabled",
        "style_embedding": "enabled" if sentence_transformer else "disabled"
    }


@app.post("/api/analyze")
async def analyze_face(file: UploadFile = File(...)):
    """얼굴 분석 및 헤어스타일 추천 (v20.1: ML 통합)"""
    start_time = time.time()
    image_hash = None

    try:
        if not GEMINI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="Gemini API 키가 설정되지 않았습니다."
            )

        if not file.filename:
            raise HTTPException(status_code=400, detail="파일명이 없습니다")

        file_ext = file.filename.lower().split('.')[-1]
        if file_ext not in ['jpg', 'jpeg', 'png', 'webp']:
            raise HTTPException(
                status_code=400,
                detail="지원하지 않는 파일 형식입니다. (jpg, jpeg, png, webp만 가능)"
            )

        logger.info(f"이미지 업로드 시작: {file.filename}")

        image_data = await file.read()
        image_hash = calculate_image_hash(image_data)

        log_structured("analysis_start", {
            "filename": file.filename,
            "file_size_kb": round(len(image_data) / 1024, 2),
            "image_hash": image_hash[:16]
        })

        # 캐시 확인
        cached_result = get_cached_result(image_hash)
        if cached_result:
            total_time = round(time.time() - start_time, 2)
            return {
                "success": True,
                "data": cached_result,
                "processing_time": total_time,
                "cached": True,
                "model_used": MODEL_NAME
            }

        # 얼굴 감지
        face_detection_start = time.time()
        face_result = detect_face(image_data)
        face_detection_time = round((time.time() - face_detection_start) * 1000, 2)

        if not face_result["has_face"]:
            log_structured("analysis_error", {
                "error_type": "no_face_detected",
                "image_hash": image_hash[:16]
            })
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "no_face_detected",
                    "message": "얼굴이 감지되지 않았습니다.\n밝은 곳에서 정면 사진을 촬영해주세요."
                }
            )

        if face_result["face_count"] > 1:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "multiple_faces",
                    "message": f"{face_result['face_count']}명의 얼굴이 감지되었습니다.\n한 명만 나온 사진을 업로드해주세요."
                }
            )

        # MediaPipe features 추출 (detect_face에서 이미 분석됨)
        mp_features = face_result.get("features", None)

        # Gemini 분석 (MediaPipe 힌트 제공)
        gemini_start = time.time()
        analysis_result = analyze_with_gemini(image_data, mp_features)
        gemini_time = round((time.time() - gemini_start) * 1000, 2)

        # ✅ ML 점수 추가
        face_shape = analysis_result.get("analysis", {}).get("face_shape")
        skin_tone = analysis_result.get("analysis", {}).get("personal_color")

        for idx, recommendation in enumerate(analysis_result.get("recommendations", []), 1):
            style_name = recommendation.get("style_name", "")

            # ML 예측 점수 계산
            ml_score = predict_ml_score(face_shape, skin_tone, style_name)

            # 결과에 추가
            recommendation['ml_confidence'] = ml_score
            recommendation['confidence_level'] = get_confidence_level(ml_score)

            # 스타일 임베딩 생성 (Sentence Transformer)
            if sentence_transformer is not None:
                try:
                    embedding = sentence_transformer.encode(style_name)
                    recommendation['style_embedding'] = embedding.tolist()
                    logger.info(f"✅ 임베딩 생성 성공: {style_name} → {len(embedding)}차원")
                except Exception as e:
                    logger.error(f"❌ 임베딩 생성 실패 ({style_name}): {str(e)}")
                    recommendation['style_embedding'] = None
            else:
                recommendation['style_embedding'] = None

            # 네이버 검색 URL
            encoded_query = urllib.parse.quote(f"{style_name} 헤어스타일")
            recommendation[
                "image_search_url"] = f"https://search.naver.com/search.naver?where=image&query={encoded_query}"

        # 캐싱
        save_to_cache(image_hash, analysis_result)

        # DB 저장
        total_time = round(time.time() - start_time, 2)
        analysis_id = save_to_database(
            image_hash=image_hash,
            analysis_result=analysis_result,
            processing_time=total_time,
            detection_method=face_result.get("method", "mediapipe"),
            mp_features=mp_features
        )

        log_structured("analysis_complete", {
            "image_hash": image_hash[:16],
            "processing_time": total_time,
            "face_detection_time_ms": face_detection_time,
            "gemini_analysis_time_ms": gemini_time,
            "mediapipe_enabled": mp_features is not None,
            "ml_enabled": ml_model is not None,
            "embedding_enabled": sentence_transformer is not None,
            "face_shape": face_shape,
            "personal_color": skin_tone,
            "analysis_id": analysis_id
        })

        return {
            "success": True,
            "data": analysis_result,
            "analysis_id": analysis_id,
            "processing_time": total_time,
            "performance": {
                "face_detection_ms": face_detection_time,
                "gemini_analysis_ms": gemini_time,
                "detection_method": face_result.get("method", "mediapipe"),
                "mediapipe_analysis": "enabled" if mp_features else "failed",
                "ml_prediction": "enabled" if ml_model else "disabled",
                "style_embedding": "enabled" if sentence_transformer else "disabled"
            },
            "cached": False,
            "model_used": MODEL_NAME
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {str(e)}")

        log_structured("analysis_error", {
            "error_type": "internal_error",
            "error_message": str(e),
            "image_hash": image_hash[:16] if image_hash else "unknown"
        })

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "internal_error",
                "message": f"분석 중 오류가 발생했습니다: {str(e)}"
            }
        )


@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """사용자 피드백 제출 엔드포인트"""
    if not SessionLocal:
        raise HTTPException(
            status_code=500,
            detail="데이터베이스 연결이 없습니다"
        )

    try:
        db = SessionLocal()

        record = db.query(AnalysisHistory).filter(
            AnalysisHistory.id == request.analysis_id
        ).first()

        if not record:
            db.close()
            raise HTTPException(
                status_code=404,
                detail=f"분석 결과를 찾을 수 없습니다 (ID: {request.analysis_id})"
            )

        feedback_column = f"style_{request.style_index}_feedback"
        clicked_column = f"style_{request.style_index}_naver_clicked"

        # 명시적 문자열 변환 (Enum → str)
        setattr(record, feedback_column, request.feedback.value)
        setattr(record, clicked_column, request.naver_clicked)
        record.feedback_at = datetime.utcnow()

        db.commit()

        logger.info(
            f"✅ 피드백 저장 성공: analysis_id={request.analysis_id}, style={request.style_index}, feedback={request.feedback}")

        log_structured("feedback_submitted", {
            "analysis_id": request.analysis_id,
            "style_index": request.style_index,
            "feedback": request.feedback,
            "naver_clicked": request.naver_clicked
        })

        db.close()

        return FeedbackResponse(
            success=True,
            message="피드백이 저장되었습니다",
            analysis_id=request.analysis_id,
            style_index=request.style_index
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 피드백 저장 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"피드백 저장 중 오류 발생: {str(e)}"
        )


@app.get("/api/stats/feedback")
async def get_feedback_stats():
    """
    피드백 통계 조회 API

    Returns:
        - total_analysis: 전체 분석 기록 수
        - total_feedback: 피드백이 있는 기록 수
        - recent_feedbacks: 최근 5개 피드백 데이터
    """
    if not SessionLocal:
        raise HTTPException(
            status_code=500,
            detail="데이터베이스 연결이 없습니다"
        )

    try:
        db = SessionLocal()

        # 전체 통계
        total = db.query(AnalysisHistory).count()
        feedback_count = db.query(AnalysisHistory).filter(
            AnalysisHistory.feedback_at.isnot(None)
        ).count()

        # 최근 5개 피드백
        recent = db.query(AnalysisHistory).filter(
            AnalysisHistory.feedback_at.isnot(None)
        ).order_by(AnalysisHistory.id.desc()).limit(5).all()

        recent_data = []
        for r in recent:
            recent_data.append({
                "id": r.id,
                "face_shape": r.face_shape,
                "personal_color": r.personal_color,
                "style_1_feedback": r.style_1_feedback,
                "style_2_feedback": r.style_2_feedback,
                "style_3_feedback": r.style_3_feedback,
                "style_1_naver_clicked": r.style_1_naver_clicked,
                "style_2_naver_clicked": r.style_2_naver_clicked,
                "style_3_naver_clicked": r.style_3_naver_clicked,
                "feedback_at": r.feedback_at.isoformat() if r.feedback_at else None,
                "created_at": r.created_at.isoformat() if r.created_at else None
            })

        # 좋아요/싫어요 통계
        like_counts = {
            "style_1": 0,
            "style_2": 0,
            "style_3": 0
        }
        dislike_counts = {
            "style_1": 0,
            "style_2": 0,
            "style_3": 0
        }

        all_feedback = db.query(AnalysisHistory).filter(
            AnalysisHistory.feedback_at.isnot(None)
        ).all()

        for record in all_feedback:
            if record.style_1_feedback == "like":
                like_counts["style_1"] += 1
            elif record.style_1_feedback == "dislike":
                dislike_counts["style_1"] += 1

            if record.style_2_feedback == "like":
                like_counts["style_2"] += 1
            elif record.style_2_feedback == "dislike":
                dislike_counts["style_2"] += 1

            if record.style_3_feedback == "like":
                like_counts["style_3"] += 1
            elif record.style_3_feedback == "dislike":
                dislike_counts["style_3"] += 1

        db.close()

        logger.info(f"📊 통계 조회: 전체 {total}개, 피드백 {feedback_count}개")

        return {
            "success": True,
            "total_analysis": total,
            "total_feedback": feedback_count,
            "like_counts": like_counts,
            "dislike_counts": dislike_counts,
            "recent_feedbacks": recent_data
        }

    except Exception as e:
        logger.error(f"❌ 통계 조회 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"통계 조회 중 오류 발생: {str(e)}"
        )


# ========== 하이브리드 추천 엔드포인트 (Gemini + ML) ==========
@app.post("/api/v2/analyze-hybrid")
async def analyze_face_hybrid(file: UploadFile = File(...)):
    """
    하이브리드 얼굴 분석 및 헤어스타일 추천 (Gemini + ML)

    플로우:
    1. MediaPipe로 얼굴형 + 피부톤 분석
    2. Gemini API로 4개 추천
    3. ML 모델로 Top-3 추천
    4. 중복 제거 후 최대 7개 반환
    """
    start_time = time.time()

    try:
        # 파일 검증
        if not file.filename:
            raise HTTPException(status_code=400, detail="파일명이 없습니다")

        file_ext = file.filename.lower().split('.')[-1]
        if file_ext not in ['jpg', 'jpeg', 'png', 'webp']:
            raise HTTPException(
                status_code=400,
                detail="지원하지 않는 파일 형식입니다."
            )

        logger.info(f"🎨 하이브리드 분석 시작: {file.filename}")

        # 이미지 읽기
        image_data = await file.read()
        image_hash = calculate_image_hash(image_data)

        # 1. MediaPipe로 얼굴 분석
        if not mediapipe_analyzer:
            raise HTTPException(
                status_code=500,
                detail="MediaPipe 분석기가 초기화되지 않았습니다."
            )

        mp_features = mediapipe_analyzer.analyze(image_data)

        if not mp_features:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "no_face_detected",
                    "message": "얼굴이 감지되지 않았습니다.\\n밝은 곳에서 정면 사진을 촬영해주세요."
                }
            )

        face_shape = mp_features.face_shape
        skin_tone = mp_features.skin_tone

        logger.info(f"✅ MediaPipe 분석: {face_shape} + {skin_tone}")

        # 2. 하이브리드 추천
        if not hybrid_service:
            raise HTTPException(
                status_code=500,
                detail="하이브리드 추천 서비스가 초기화되지 않았습니다."
            )

        recommendation_result = hybrid_service.recommend(
            image_data, face_shape, skin_tone
        )

        # 3. 네이버 검색 URL 추가
        import urllib.parse
        for rec in recommendation_result.get("recommendations", []):
            style_name = rec.get("style_name", "")
            encoded_query = urllib.parse.quote(f"{style_name} 헤어스타일")
            rec["image_search_url"] = f"https://search.naver.com/search.naver?where=image&query={encoded_query}"

        # 4. DB에 분석 결과 저장
        total_time = round(time.time() - start_time, 2)
        analysis_id = None

        if SessionLocal:
            try:
                db = SessionLocal()

                # AnalysisHistory 레코드 생성
                new_record = AnalysisHistory(
                    user_id="anonymous",
                    image_hash=image_hash,
                    face_shape=face_shape,
                    personal_color=skin_tone,
                    recommendations=recommendation_result,
                    processing_time=total_time,
                    detection_method="hybrid",
                    recommended_styles=recommendation_result.get("recommendations", [])
                )

                db.add(new_record)
                db.commit()
                db.refresh(new_record)

                analysis_id = new_record.id

                logger.info(f"✅ 분석 결과 DB 저장 완료: analysis_id={analysis_id}")

                db.close()
            except Exception as e:
                logger.error(f"❌ DB 저장 실패: {str(e)}")
                # DB 저장 실패해도 분석 결과는 반환

        logger.info(f"✅ 하이브리드 분석 완료 ({total_time}초)")

        return {
            "success": True,
            "analysis_id": analysis_id,
            "data": recommendation_result,
            "processing_time": total_time,
            "method": "hybrid",
            "mediapipe_features": {
                "face_shape": face_shape,
                "skin_tone": skin_tone,
                "confidence": mp_features.confidence
            },
            "model_used": "gemini-1.5-flash-latest + hairstyle_recommender.pt"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 하이브리드 분석 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"분석 중 오류가 발생했습니다: {str(e)}"
        )


@app.post("/api/v2/feedback")
async def collect_feedback(
    face_shape: str,
    skin_tone: str,
    hairstyle_id: int,
    user_reaction: str,
    ml_prediction: float,
    user_id: str = "anonymous"
):
    """
    사용자 피드백 수집 엔드포인트 (v2)

    Args:
        face_shape: 얼굴형 ("계란형", "둥근형", "긴형", "각진형")
        skin_tone: 피부톤 ("가을웜", "겨울쿨", "봄웜", "여름쿨")
        hairstyle_id: 헤어스타일 ID (0-based index)
        user_reaction: "👍" (좋아요) or "👎" (싫어요)
        ml_prediction: ML 모델 예측 점수
        user_id: 사용자 ID (기본값: "anonymous")

    Returns:
        {"total_feedbacks": int, "retrain_triggered": bool, "retrain_job_id": str}

    Ground Truth Rules:
        👍 -> 90.0 (user LIKED this combination)
        👎 -> 10.0 (user DISLIKED this combination)
    """
    if not feedback_collector:
        raise HTTPException(
            status_code=500,
            detail="피드백 수집기가 초기화되지 않았습니다."
        )

    try:
        # 입력 검증
        if user_reaction not in ["👍", "👎"]:
            raise HTTPException(
                status_code=400,
                detail="user_reaction은 '👍' 또는 '👎'만 가능합니다."
            )

        # 피드백 저장
        result = feedback_collector.save_feedback(
            face_shape=face_shape,
            skin_tone=skin_tone,
            hairstyle_id=hairstyle_id,
            user_reaction=user_reaction,
            ml_prediction=ml_prediction,
            user_id=user_id
        )

        retrain_job_id = None

        # 재학습 트리거 확인
        if result['retrain_triggered'] and retrain_queue:
            # 재학습 작업을 큐에 추가
            job = retrain_queue.add_job(result['total_feedbacks'])
            retrain_job_id = job['job_id']

            logger.info(
                f"🔄 재학습 작업 생성: {retrain_job_id} "
                f"(피드백 {result['total_feedbacks']}개)"
            )

            log_structured("retrain_job_created", {
                "job_id": retrain_job_id,
                "feedback_count": result['total_feedbacks']
            })

        logger.info(
            f"✅ 피드백 수집 완료: {face_shape} + {skin_tone} + ID#{hairstyle_id} "
            f"-> {user_reaction} | Total: {result['total_feedbacks']}"
        )

        log_structured("feedback_collected", {
            "face_shape": face_shape,
            "skin_tone": skin_tone,
            "hairstyle_id": hairstyle_id,
            "user_reaction": user_reaction,
            "ml_prediction": ml_prediction,
            "total_feedbacks": result['total_feedbacks'],
            "retrain_triggered": result['retrain_triggered'],
            "retrain_job_id": retrain_job_id
        })

        return {
            "success": True,
            "total_feedbacks": result['total_feedbacks'],
            "retrain_triggered": result['retrain_triggered'],
            "retrain_job_id": retrain_job_id,
            "message": "피드백이 성공적으로 저장되었습니다."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 피드백 수집 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"피드백 수집 중 오류가 발생했습니다: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)