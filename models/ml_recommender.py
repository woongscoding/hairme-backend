"""
ML 기반 독립형 헤어스타일 추천기

MediaPipe 분석 결과 (얼굴형 + 피부톤)로 학습된 ML 모델을 사용해
모든 헤어스타일의 추천 점수를 예측하고 Top-K를 반환

Author: HairMe ML Team
Date: 2025-11-08
Version: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple
import logging
import sys

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.style_preprocessor import normalize_style_name

logger = logging.getLogger(__name__)


class RecommendationModel(nn.Module):
    """PyTorch 추천 모델 (동일한 구조)"""

    def __init__(self, input_dim: int = 392):
        super(RecommendationModel, self).__init__()

        hidden_dims = [256, 128, 64]
        dropout_rates = [0.3, 0.2, 0.0]

        layers = []
        prev_dim = input_dim

        for hidden_dim, dropout_rate in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MLHairstyleRecommender:
    """ML 기반 헤어스타일 추천기"""

    # MediaPipe와 호환되는 카테고리
    FACE_SHAPES = ["각진형", "둥근형", "긴형", "계란형"]
    SKIN_TONES = ["겨울쿨", "가을웜", "봄웜", "여름쿨"]

    def __init__(
        self,
        model_path: str = "models/hairstyle_recommender.pt",
        embeddings_path: str = "data_source/style_embeddings.npz"
    ):
        """
        초기화

        Args:
            model_path: 학습된 모델 경로
            embeddings_path: 헤어스타일 임베딩 경로
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. 모델 로드
        logger.info(f"📂 ML 모델 로딩: {model_path}")
        self.model = RecommendationModel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()  # 추론 모드
        logger.info(f"✅ 모델 로드 완료 (디바이스: {self.device})")

        # 2. 헤어스타일 임베딩 로드
        logger.info(f"📂 임베딩 로딩: {embeddings_path}")
        data = np.load(embeddings_path, allow_pickle=True)
        self.styles = data['styles'].tolist()  # 헤어스타일명 리스트
        self.embeddings = data['embeddings']  # (N, 384) 임베딩
        logger.info(f"✅ 임베딩 로드 완료: {len(self.styles)}개 스타일")

        # 스타일명 -> 인덱스 매핑
        self.style_to_idx = {style: idx for idx, style in enumerate(self.styles)}

    def _encode_face_shape(self, face_shape: str) -> np.ndarray:
        """얼굴형을 one-hot 인코딩"""
        vec = np.zeros(4, dtype=np.float32)

        # 하트형은 계란형으로 매핑
        if face_shape == "하트형":
            face_shape = "계란형"
            logger.debug("하트형을 계란형으로 매핑")

        if face_shape in self.FACE_SHAPES:
            idx = self.FACE_SHAPES.index(face_shape)
            vec[idx] = 1.0
        else:
            logger.warning(f"알 수 없는 얼굴형: {face_shape}, 계란형으로 기본값 사용")
            vec[3] = 1.0  # 계란형

        return vec

    def _encode_skin_tone(self, skin_tone: str) -> np.ndarray:
        """피부톤을 one-hot 인코딩"""
        vec = np.zeros(4, dtype=np.float32)

        if skin_tone in self.SKIN_TONES:
            idx = self.SKIN_TONES.index(skin_tone)
            vec[idx] = 1.0
        else:
            logger.warning(f"알 수 없는 피부톤: {skin_tone}, 봄웜으로 기본값 사용")
            vec[2] = 1.0  # 봄웜

        return vec

    def _create_feature_vector(
        self,
        face_shape: str,
        skin_tone: str,
        style_embedding: np.ndarray
    ) -> np.ndarray:
        """
        특징 벡터 생성 (392차원)

        Args:
            face_shape: 얼굴형
            skin_tone: 피부톤
            style_embedding: 헤어스타일 임베딩 (384차원)

        Returns:
            특징 벡터 (392차원)
        """
        face_vec = self._encode_face_shape(face_shape)
        tone_vec = self._encode_skin_tone(skin_tone)

        # 연결: [face(4) + tone(4) + style(384)] = 392
        feature = np.concatenate([face_vec, tone_vec, style_embedding])

        return feature

    def predict_score(
        self,
        face_shape: str,
        skin_tone: str,
        hairstyle: str
    ) -> float:
        """
        특정 헤어스타일의 추천 점수 예측 (띄어쓰기 정규화 적용)

        Args:
            face_shape: 얼굴형
            skin_tone: 피부톤
            hairstyle: 헤어스타일명

        Returns:
            추천 점수 (0-100)
        """
        # 띄어쓰기 정규화 적용
        normalized_style = normalize_style_name(hairstyle)

        if normalized_style not in self.style_to_idx:
            logger.warning(f"미등록 헤어스타일: '{hairstyle}' (정규화: '{normalized_style}')")
            return 0.0

        # 임베딩 가져오기 (정규화된 스타일명 사용)
        idx = self.style_to_idx[normalized_style]
        style_embedding = self.embeddings[idx]

        # 특징 벡터 생성
        feature = self._create_feature_vector(face_shape, skin_tone, style_embedding)

        # 모델 추론
        with torch.no_grad():
            feature_tensor = torch.FloatTensor(feature).unsqueeze(0).to(self.device)
            score_tensor = self.model(feature_tensor)
            score = score_tensor.cpu().item()

        # 0-100 범위로 클리핑
        score = max(0.0, min(100.0, score))

        return round(score, 2)

    def recommend_top_k(
        self,
        face_shape: str,
        skin_tone: str,
        k: int = 3
    ) -> List[Dict[str, any]]:
        """
        Top-K 헤어스타일 추천

        Args:
            face_shape: 얼굴형 (예: "계란형")
            skin_tone: 피부톤 (예: "봄웜")
            k: 추천 개수

        Returns:
            추천 리스트 [{"hairstyle": "...", "score": 85.3}, ...]
        """
        logger.info(f"🤖 ML 추천 시작: {face_shape} + {skin_tone} (Top-{k})")

        # 모든 헤어스타일에 대해 점수 예측
        all_scores = []

        # 배치 처리로 최적화
        batch_size = 64
        num_styles = len(self.styles)

        for i in range(0, num_styles, batch_size):
            batch_end = min(i + batch_size, num_styles)
            batch_embeddings = self.embeddings[i:batch_end]

            # 배치 특징 생성
            batch_features = []
            for embedding in batch_embeddings:
                feature = self._create_feature_vector(face_shape, skin_tone, embedding)
                batch_features.append(feature)

            batch_features = np.array(batch_features, dtype=np.float32)

            # 배치 추론
            with torch.no_grad():
                batch_tensor = torch.FloatTensor(batch_features).to(self.device)
                scores_tensor = self.model(batch_tensor)
                scores = scores_tensor.cpu().numpy().flatten()

            # 결과 저장
            for j, score in enumerate(scores):
                style_idx = i + j
                all_scores.append({
                    "hairstyle": self.styles[style_idx],
                    "score": max(0.0, min(100.0, float(score)))
                })

        # 점수 기준 정렬
        all_scores.sort(key=lambda x: x['score'], reverse=True)

        # Top-K 추출
        top_k_recommendations = all_scores[:k]

        # 점수 반올림
        for rec in top_k_recommendations:
            rec['score'] = round(rec['score'], 2)

        logger.info(
            f"✅ ML 추천 완료: {[r['hairstyle'] for r in top_k_recommendations]}"
        )

        return top_k_recommendations

    def batch_predict(
        self,
        face_shape: str,
        skin_tone: str,
        hairstyles: List[str]
    ) -> Dict[str, float]:
        """
        여러 헤어스타일의 점수를 한 번에 예측 (띄어쓰기 정규화 적용)

        Args:
            face_shape: 얼굴형
            skin_tone: 피부톤
            hairstyles: 헤어스타일 리스트

        Returns:
            {헤어스타일: 점수} 딕셔너리
        """
        results = {}

        # 유효한 스타일만 필터링 (정규화 후 확인)
        valid_styles = []
        style_mapping = {}  # 원본 -> 정규화 매핑
        for s in hairstyles:
            normalized = normalize_style_name(s)
            if normalized in self.style_to_idx:
                valid_styles.append(s)
                style_mapping[s] = normalized

        if not valid_styles:
            logger.warning("유효한 헤어스타일이 없습니다")
            return results

        # 배치 특징 생성 (정규화된 스타일명 사용)
        batch_features = []
        for style in valid_styles:
            normalized = style_mapping[style]
            idx = self.style_to_idx[normalized]
            embedding = self.embeddings[idx]
            feature = self._create_feature_vector(face_shape, skin_tone, embedding)
            batch_features.append(feature)

        batch_features = np.array(batch_features, dtype=np.float32)

        # 배치 추론
        with torch.no_grad():
            batch_tensor = torch.FloatTensor(batch_features).to(self.device)
            scores_tensor = self.model(batch_tensor)
            scores = scores_tensor.cpu().numpy().flatten()

        # 결과 저장
        for style, score in zip(valid_styles, scores):
            results[style] = round(max(0.0, min(100.0, float(score))), 2)

        return results


# ========== 싱글톤 인스턴스 (전역 사용) ==========
_recommender_instance = None


def get_ml_recommender() -> MLHairstyleRecommender:
    """
    ML 추천기 싱글톤 인스턴스 가져오기

    Returns:
        MLHairstyleRecommender 인스턴스
    """
    global _recommender_instance

    if _recommender_instance is None:
        logger.info("🔧 ML 추천기 초기화 중...")
        _recommender_instance = MLHairstyleRecommender()
        logger.info("✅ ML 추천기 준비 완료")

    return _recommender_instance
