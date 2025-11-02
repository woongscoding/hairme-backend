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

# ========== OpenCV 얼굴 분석 모듈 임포트 ==========
from models.face_analyzer import extract_face_features, create_enhanced_prompt, FaceFeatures

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

    # OpenCV 측정 데이터
    opencv_face_ratio = Column(Float)
    opencv_forehead_ratio = Column(Float)
    opencv_cheekbone_ratio = Column(Float)
    opencv_jaw_ratio = Column(Float)
    opencv_prediction = Column(String(50))
    opencv_confidence = Column(Float)
    opencv_gemini_agreement = Column(Boolean)

    # v20: 추천 스타일 저장
    recommended_styles = Column(JSON)

    # v20: 피드백 컬럼
    style_1_feedback = Column(SQLEnum(FeedbackType), nullable=True)
    style_2_feedback = Column(SQLEnum(FeedbackType), nullable=True)
    style_3_feedback = Column(SQLEnum(FeedbackType), nullable=True)

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
    description="AI 기반 헤어스타일 추천 서비스 (v20.1: ML 통합)",
    version="20.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 앱 시작 이벤트 ==========
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 ML 모델 로드"""
    logger.info("🚀 서버 시작 중...")

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


# ========== 데이터베이스 연결 ==========
SessionLocal = None
DATABASE_URL = os.getenv("DATABASE_URL")
DB_PASSWORD = os.getenv("DB_PASSWORD")

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

# ========== OpenCV 얼굴 감지기 ==========
face_cascade = None
try:
    cascade_paths = [
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        '/usr/local/lib/python3.11/site-packages/cv2/data/haarcascade_frontalface_default.xml',
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
    ]

    for path in cascade_paths:
        if os.path.exists(path):
            face_cascade = cv2.CascadeClassifier(path)
            if not face_cascade.empty():
                logger.info(f"✅ OpenCV 얼굴 감지기 초기화 완료: {path}")
                break

    if face_cascade is None or face_cascade.empty():
        logger.error("❌ OpenCV 얼굴 감지기 초기화 실패")
        face_cascade = None

except Exception as e:
    logger.error(f"OpenCV 얼굴 감지기 초기화 실패: {str(e)}")
    face_cascade = None

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
    """얼굴 감지 (OpenCV 우선, 실패 시 Gemini)"""
    if face_cascade is not None and not face_cascade.empty():
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(100, 100)
                )

                if len(faces) > 0:
                    log_structured("face_detection", {
                        "method": "opencv",
                        "face_count": len(faces),
                        "success": True
                    })
                    return {
                        "has_face": True,
                        "face_count": len(faces),
                        "method": "opencv"
                    }

        except Exception as e:
            logger.warning(f"OpenCV 얼굴 감지 실패: {str(e)}")

    logger.info("OpenCV 실패, Gemini로 얼굴 검증 시작...")
    gemini_result = verify_face_with_gemini(image_data)

    log_structured("face_detection", {
        "method": "gemini",
        "face_count": gemini_result.get("face_count", 0),
        "success": gemini_result.get("has_face", False)
    })

    return gemini_result


def analyze_with_gemini(image_data: bytes) -> dict:
    """Gemini Vision API로 얼굴 분석"""
    try:
        image = Image.open(io.BytesIO(image_data))

        opencv_features = extract_face_features(image_data)

        if opencv_features:
            prompt = create_enhanced_prompt(opencv_features)
            logger.info(f"✅ OpenCV 힌트 적용: {opencv_features.face_shape_hint}")
        else:
            prompt = ANALYSIS_PROMPT
            logger.warning("⚠️ OpenCV 특징 추출 실패, 기본 프롬프트 사용")

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
        opencv_features: Optional[FaceFeatures] = None
) -> Optional[int]:  # ✅ 반환 타입 추가
    """분석 결과를 MySQL에 저장하고 ID 반환"""
    if not SessionLocal:
        logger.warning("⚠️ 데이터베이스 연결이 없어 저장을 생략합니다.")
        return None  # ✅ None 반환

    try:
        db = SessionLocal()

        gemini_shape = analysis_result.get("analysis", {}).get("face_shape")

        opencv_agreement = None
        if opencv_features:
            opencv_agreement = (
                    opencv_features.face_shape_hint in gemini_shape or
                    gemini_shape in opencv_features.face_shape_hint
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
            opencv_face_ratio=opencv_features.face_ratio if opencv_features else None,
            opencv_forehead_ratio=opencv_features.forehead_ratio if opencv_features else None,
            opencv_cheekbone_ratio=opencv_features.cheekbone_ratio if opencv_features else None,
            opencv_jaw_ratio=opencv_features.jaw_ratio if opencv_features else None,
            opencv_prediction=opencv_features.face_shape_hint if opencv_features else None,
            opencv_confidence=opencv_features.confidence if opencv_features else None,
            opencv_gemini_agreement=opencv_agreement
        )

        db.add(history)
        db.commit()
        db.refresh(history)

        logger.info(f"✅ DB 저장 성공 (ID: {history.id})")
        log_structured("database_saved", {
            "record_id": history.id,
            "opencv_enabled": opencv_features is not None,
            "agreement": opencv_agreement,
            "recommendations_count": len(recommendations)
        })

        db.close()
        return history.id  # ✅ ID 반환 추가!

    except Exception as e:
        logger.error(f"❌ DB 저장 실패: {str(e)}")
        return None  # ✅ 에러 시 None 반환

# ========== API 엔드포인트 ==========
@app.get("/")
async def root():
    """Root 엔드포인트"""
    face_detection_status = "enabled" if (face_cascade is not None and not face_cascade.empty()) else "disabled"
    return {
        "message": "헤어스타일 분석 API - v20.1 (ML 통합)",
        "version": "20.1.0",
        "model": MODEL_NAME,
        "status": "running",
        "features": {
            "face_detection": face_detection_status,
            "opencv_analysis": "enabled",
            "gemini_analysis": "enabled" if GEMINI_API_KEY else "disabled",
            "redis_cache": "enabled" if redis_client else "disabled",
            "database": "enabled" if SessionLocal else "disabled",
            "feedback_system": "enabled",
            "ml_prediction": "enabled" if ml_model else "disabled"
        }
    }


@app.get("/api/health")
async def health_check():
    """헬스체크 엔드포인트"""
    face_detection_status = "enabled" if (face_cascade is not None and not face_cascade.empty()) else "disabled"

    return {
        "status": "healthy",
        "version": "20.1.0",
        "model": MODEL_NAME,
        "face_detection": face_detection_status,
        "opencv_analysis": "enabled",
        "gemini_api": "configured" if GEMINI_API_KEY else "not_configured",
        "redis": "connected" if redis_client else "disconnected",
        "database": "connected" if SessionLocal else "disconnected",
        "feedback_system": "enabled",
        "ml_model": "enabled" if ml_model else "disabled"
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

        # Gemini 분석
        gemini_start = time.time()
        analysis_result = analyze_with_gemini(image_data)
        gemini_time = round((time.time() - gemini_start) * 1000, 2)

        opencv_features = extract_face_features(image_data)

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

            # 네이버 검색 URL
            encoded_query = urllib.parse.quote(f"{style_name} 헤어스타일")
            recommendation[
                "image_search_url"] = f"https://search.naver.com/search.naver?where=image&query={encoded_query}"

        # 캐싱
        save_to_cache(image_hash, analysis_result)

        # DB 저장
        total_time = round(time.time() - start_time, 2)
        analysis_id = save_to_database(  # ✅ 이 부분 추가!
            image_hash=image_hash,
            analysis_result=analysis_result,
            processing_time=total_time,
            detection_method=face_result.get("method", "opencv"),
            opencv_features=opencv_features
        )

        log_structured("analysis_complete", {
            "image_hash": image_hash[:16],
            "processing_time": total_time,
            "face_detection_time_ms": face_detection_time,
            "gemini_analysis_time_ms": gemini_time,
            "opencv_enabled": opencv_features is not None,
            "ml_enabled": ml_model is not None,
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
                "detection_method": face_result.get("method", "opencv"),
                "opencv_analysis": "enabled" if opencv_features else "failed",
                "ml_prediction": "enabled" if ml_model else "disabled"
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

        setattr(record, feedback_column, request.feedback)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)