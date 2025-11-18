"""ML model loading and prediction utilities"""

import os
import pickle
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from config.settings import settings
from core.logging import logger, log_structured


# ========== ML Model Definition ==========
class HairstyleRecommender(nn.Module):
    """Hairstyle recommendation ML model"""

    def __init__(self, n_faces: int = 5, n_skins: int = 3, n_styles: int = 6,
                 emb_dim: int = 16, hidden_dim: int = 64):
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

    def forward(self, face: torch.Tensor, skin: torch.Tensor, style: torch.Tensor):
        face_emb = self.face_emb(face)
        skin_emb = self.skin_emb(skin)
        style_emb = self.style_emb(style)

        x = torch.cat([face_emb, skin_emb, style_emb], dim=1)
        shared = self.shared_layers(x)

        score_pred = self.score_head(shared).squeeze(-1)
        feedback_logits = self.feedback_head(shared)

        return score_pred, feedback_logits


# ========== Global Variables (Model & Encoders) ==========
ml_model: Optional[HairstyleRecommender] = None
face_encoder: Optional[Any] = None
skin_encoder: Optional[Any] = None
style_encoder: Optional[Any] = None
sentence_transformer: Optional[Any] = None


# ========== Model Loaders ==========
def load_ml_model() -> bool:
    """
    Load ML model and encoders

    Returns:
        bool: True if successful, False otherwise
    """
    global ml_model, face_encoder, skin_encoder, style_encoder

    try:
        model_path = settings.ML_MODEL_PATH
        encoder_path = settings.ML_ENCODER_PATH

        # Check file existence
        if not os.path.exists(model_path):
            logger.warning(f"⚠️ 모델 파일 없음: {model_path}")
            return False

        if not os.path.exists(encoder_path):
            logger.warning(f"⚠️ 인코더 파일 없음: {encoder_path}")
            return False

        # Load model
        ml_model = HairstyleRecommender()
        ml_model.load_state_dict(torch.load(
            model_path,
            map_location=torch.device('cpu'),
            weights_only=True
        ))
        ml_model.eval()

        # Load encoders
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


def load_sentence_transformer() -> bool:
    """
    Load Sentence Transformer model (for hairstyle embedding)

    Returns:
        bool: True if successful, False otherwise
    """
    global sentence_transformer

    try:
        from sentence_transformers import SentenceTransformer

        model_name = settings.SENTENCE_TRANSFORMER_MODEL
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


# ========== Prediction Functions ==========
def predict_ml_score(face_shape: str, skin_tone: str, hairstyle: str) -> float:
    """
    Predict score using ML model

    Args:
        face_shape: Face shape (e.g., "계란형")
        skin_tone: Skin tone (e.g., "봄웜")
        hairstyle: Hairstyle (e.g., "시스루뱅 단발")

    Returns:
        Predicted score (0.0 ~ 1.0)
    """
    if ml_model is None:
        return 0.85  # Default value if model not loaded

    try:
        # ========== Gemini output → ML input mapping ==========
        skin_tone_mapping: Dict[str, str] = {
            "봄웜": "웜톤",
            "가을웜": "웜톤",
            "여름쿨": "쿨톤",
            "겨울쿨": "쿨톤"
        }

        mapped_skin = skin_tone_mapping.get(skin_tone, "중간톤")

        # ========== Encoding ==========
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

        # ========== Tensor conversion ==========
        face_tensor = torch.tensor([face_encoded], dtype=torch.long)
        skin_tensor = torch.tensor([skin_encoded], dtype=torch.long)
        style_tensor = torch.tensor([style_encoded], dtype=torch.long)

        # ========== Prediction ==========
        with torch.no_grad():
            score_pred, _ = ml_model(face_tensor, skin_tensor, style_tensor)
            score = score_pred.item()

        logger.info(f"ML 예측: {face_shape} + {mapped_skin} + {hairstyle} → {score:.3f}")
        return round(score, 3)

    except Exception as e:
        logger.error(f"ML 예측 실패: {str(e)}")
        return 0.85


def get_confidence_level(score: float) -> str:
    """
    Convert score to confidence level

    Args:
        score: Confidence score (0.0 ~ 1.0)

    Returns:
        Confidence level string
    """
    if score >= settings.CONFIDENCE_THRESHOLD_VERY_HIGH:
        return "매우 높음"
    elif score >= settings.CONFIDENCE_THRESHOLD_HIGH:
        return "높음"
    elif score >= settings.CONFIDENCE_THRESHOLD_MEDIUM:
        return "보통"
    else:
        return "낮음"
