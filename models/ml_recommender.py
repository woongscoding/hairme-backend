"""
ML 기반 독립형 헤어스타일 추천기

MediaPipe 분석 결과 (얼굴형 + 피부톤)로 학습된 ML 모델을 사용해
모든 헤어스타일의 추천 점수를 예측하고 Top-K를 반환

Author: HairMe ML Team
Date: 2025-11-08
Version: 1.2.0 (Normalized Label Support - v5)

v1.2.0 변경사항:
- 라벨 정규화 모델(v5) 지원 추가
- 모델 출력 역변환 로직 (0~1 → 10~95)
- Sigmoid 출력층 지원
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import logging
import sys
from difflib import SequenceMatcher

# TYPE_CHECKING을 사용하여 런타임에는 import하지 않음
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.style_preprocessor import normalize_style_name

logger = logging.getLogger(__name__)

# ========== 라벨 정규화 상수 (v5 모델용) ==========
LABEL_MIN = 10.0  # 원본 점수 최소값
LABEL_MAX = 95.0  # 원본 점수 최대값
LABEL_RANGE = LABEL_MAX - LABEL_MIN  # 85


# ========== 학습 데이터 특징 통계 (입력 스케일링용) ==========
# ai_face_1000.npz에서 추출한 통계 (5910 샘플)
FACE_FEATURE_STATS = {
    0: {"min": 0.99, "max": 1.51, "mean": 1.20, "std": 0.06},  # face_ratio
    1: {
        "min": 301.10,
        "max": 495.30,
        "mean": 458.13,
        "std": 14.31,
    },  # forehead_width (pixel)
    2: {
        "min": 421.40,
        "max": 641.00,
        "mean": 561.34,
        "std": 19.73,
    },  # cheekbone_width (pixel)
    3: {
        "min": 333.90,
        "max": 524.10,
        "mean": 447.70,
        "std": 19.82,
    },  # jaw_width (pixel)
    4: {"min": 0.71, "max": 0.89, "mean": 0.82, "std": 0.02},  # forehead_ratio
    5: {"min": 0.73, "max": 0.86, "mean": 0.80, "std": 0.02},  # jaw_ratio
}

SKIN_FEATURE_STATS = {
    0: {"min": 50.53, "max": 89.26, "mean": 79.91, "std": 3.90},  # ITA_value
    1: {"min": 5.96, "max": 142.39, "mean": 12.09, "std": 10.97},  # hue_value
}


def denormalize_score(normalized: float) -> float:
    """0~1 점수를 원본 스케일로 역변환 (10~95)"""
    return normalized * LABEL_RANGE + LABEL_MIN


def scale_input_features(face_features: np.ndarray, skin_features: np.ndarray) -> tuple:
    """
    추론 입력을 학습 데이터 분포에 맞게 스케일링

    문제:
    - 학습 데이터: 얼굴 너비 300-600 픽셀 (고해상도 이미지)
    - 실제 추론: 얼굴 너비 70-150 픽셀 (다양한 해상도)
    - 이 스케일 불일치로 인해 OOD(Out-of-Distribution) 예측 발생

    해결책:
    - 픽셀 기반 특징(forehead, cheekbone, jaw width)을 학습 데이터 평균 스케일로 변환
    - 비율 기반 특징(face_ratio, forehead_ratio, jaw_ratio)은 스케일 불변이므로 그대로 유지

    Args:
        face_features: [face_ratio, forehead_width, cheekbone_width, jaw_width, forehead_ratio, jaw_ratio]
        skin_features: [ITA_value, hue_value]

    Returns:
        (scaled_face_features, scaled_skin_features)
    """
    face_scaled = face_features.copy()
    skin_scaled = skin_features.copy()

    # 얼굴 특징 스케일링
    # - 인덱스 0, 4, 5: 비율 특징 (스케일 불변) - 스케일링 필요 없음
    # - 인덱스 1, 2, 3: 픽셀 너비 특징 (스케일 의존) - 스케일링 필요

    # 입력 이미지의 스케일 추정 (cheekbone_width 기준)
    input_cheekbone = face_features[2]
    train_cheekbone_mean = FACE_FEATURE_STATS[2]["mean"]  # 561.34

    # 스케일 팩터 계산 (입력을 학습 데이터 스케일로 변환)
    if input_cheekbone > 0:
        scale_factor = train_cheekbone_mean / input_cheekbone
    else:
        scale_factor = 1.0

    # 픽셀 기반 특징만 스케일링 (인덱스 1, 2, 3)
    face_scaled[1] = face_features[1] * scale_factor  # forehead_width
    face_scaled[2] = face_features[2] * scale_factor  # cheekbone_width
    face_scaled[3] = face_features[3] * scale_factor  # jaw_width

    # 스케일링된 값이 학습 데이터 범위 내에 있도록 클리핑
    for idx in [1, 2, 3]:
        min_val = FACE_FEATURE_STATS[idx]["min"]
        max_val = FACE_FEATURE_STATS[idx]["max"]
        face_scaled[idx] = np.clip(face_scaled[idx], min_val, max_val)

    # 비율 특징도 학습 데이터 범위 내에 있도록 클리핑
    for idx in [0, 4, 5]:
        min_val = FACE_FEATURE_STATS[idx]["min"]
        max_val = FACE_FEATURE_STATS[idx]["max"]
        face_scaled[idx] = np.clip(face_scaled[idx], min_val, max_val)

    # 피부 특징 클리핑 (이미 스케일 불변)
    for idx in [0, 1]:
        min_val = SKIN_FEATURE_STATS[idx]["min"]
        max_val = SKIN_FEATURE_STATS[idx]["max"]
        skin_scaled[idx] = np.clip(skin_scaled[idx], min_val, max_val)

    return face_scaled, skin_scaled


class AttentionLayer(nn.Module):
    """
    Multi-head self-attention layer for multi-token inputs

    Note: 이 레이어는 여러 토큰 간의 상호작용을 학습합니다.
    단일 토큰에 적용하면 identity 연산이 되므로, 최소 2개 이상의 토큰이 필요합니다.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention across multiple tokens
        attn_out, _ = self.attention(x, x, x)
        # Residual connection + layer norm
        x = self.norm(x + self.dropout(attn_out))
        return x


class MultiTokenAttentionLayer(nn.Module):
    """
    3-Token Cross-Attention Layer

    face_proj, skin_proj, style_emb를 3개의 개별 토큰으로 구성하여
    토큰 간의 상호작용을 학습합니다.

    구조:
    - Token 1: face_proj (projected to token_dim)
    - Token 2: skin_proj (projected to token_dim)
    - Token 3: style_emb (projected to token_dim)
    - Self-attention으로 3개 토큰 간 관계 학습
    - 3개 토큰을 concat하여 출력
    """

    def __init__(
        self,
        face_dim: int = 64,
        skin_dim: int = 32,
        style_dim: int = 384,
        token_dim: int = 128,  # 통일된 토큰 차원
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_dim = token_dim

        # 각 입력을 동일한 token_dim으로 projection
        self.face_to_token = nn.Linear(face_dim, token_dim)
        self.skin_to_token = nn.Linear(skin_dim, token_dim)
        self.style_to_token = nn.Linear(style_dim, token_dim)

        # Multi-head self-attention (3 tokens)
        self.attention = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(token_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.norm2 = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        face_proj: torch.Tensor,  # (batch, 64)
        skin_proj: torch.Tensor,  # (batch, 32)
        style_emb: torch.Tensor,  # (batch, 384)
    ) -> torch.Tensor:
        """
        Returns:
            (batch, token_dim * 3) - 3개 토큰의 concat 결과
        """
        batch_size = face_proj.size(0)

        # 각 특징을 token_dim으로 projection
        face_token = self.face_to_token(face_proj)  # (batch, token_dim)
        skin_token = self.skin_to_token(skin_proj)  # (batch, token_dim)
        style_token = self.style_to_token(style_emb)  # (batch, token_dim)

        # 3개 토큰으로 시퀀스 구성: (batch, 3, token_dim)
        tokens = torch.stack([face_token, skin_token, style_token], dim=1)

        # Self-attention across 3 tokens
        attn_out, _ = self.attention(tokens, tokens, tokens)
        tokens = self.norm1(tokens + self.dropout(attn_out))

        # Feed-forward network
        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + self.dropout(ffn_out))

        # 3개 토큰을 flatten하여 반환: (batch, token_dim * 3)
        output = tokens.reshape(batch_size, -1)

        return output


class RecommendationModel(nn.Module):
    """
    연속형 변수 기반 추천 모델 v4

    입력:
    - face_features: [batch, 6] - MediaPipe 얼굴 측정값
    - skin_features: [batch, 2] - MediaPipe 피부 측정값
    - style_emb: [batch, 384] - 헤어스타일 임베딩

    Note:
    - use_attention=True는 기존 체크포인트 호환성을 위해 유지
    - 단일 토큰 attention은 실질적으로 LayerNorm+Dropout과 동일 (identity에 가까움)
    - 새로운 학습 시에는 RecommendationModelV6 사용 권장
    """

    def __init__(
        self,
        face_feat_dim: int = 6,
        skin_feat_dim: int = 2,
        style_embed_dim: int = 384,
        use_attention: bool = True,  # 기존 체크포인트 호환성 유지
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.face_feat_dim = face_feat_dim
        self.skin_feat_dim = skin_feat_dim
        self.style_embed_dim = style_embed_dim

        # Input projection layers
        self.face_projection = nn.Sequential(
            nn.Linear(face_feat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        self.skin_projection = nn.Sequential(
            nn.Linear(skin_feat_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        # Total dimension after projection
        self.total_dim = 64 + 32 + style_embed_dim  # 96 + 384 = 480

        # Attention layer (단일 토큰 - 실질적으로 LayerNorm+Dropout)
        # WARNING: 이 구조는 의미 없는 attention 적용임
        # 새로운 모델에서는 MultiTokenAttentionLayer 사용 권장
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionLayer(
                embed_dim=self.total_dim, num_heads=8, dropout=0.1
            )

        # Feature fusion network
        self.fc1 = nn.Linear(self.total_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate * 0.7)

        # Residual connection
        self.residual_proj = nn.Linear(self.total_dim, 128)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout_rate * 0.5)

        self.fc4 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)

    def forward(
        self,
        face_features: torch.Tensor,
        skin_features: torch.Tensor,
        style_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass"""
        # Project features
        face_proj = self.face_projection(face_features)
        skin_proj = self.skin_projection(skin_features)

        # Concatenate all features
        x = torch.cat([face_proj, skin_proj, style_emb], dim=1)

        # Apply attention if enabled (단일 토큰이므로 실질적으로 LayerNorm)
        if self.use_attention:
            x_att = x.unsqueeze(1)
            x_att = self.attention(x_att)
            x = x_att.squeeze(1)

        # Store for residual
        residual = self.residual_proj(x)

        # Main network
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        # Add residual connection
        x = x + residual

        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = torch.relu(x)

        x = self.fc_out(x)

        # 스케일링 적용 (학습 시 30~90점 범위)
        # 클램핑 제거 - 원본 점수를 유지하여 Top-K 내에서 Min-Max 정규화 가능하게 함
        x = (x - 29.0) * 7.5 + 60.0
        # 참고: 클램핑은 recommend_top_k에서 Min-Max 정규화 후 적용

        return x.squeeze(-1)


class NormalizedRecommendationModel(nn.Module):
    """
    정규화된 라벨 기반 추천 모델 v5

    핵심 특징:
    - 출력층에 Sigmoid 활성화 함수 사용 (0~1 출력 보장)
    - 추론 시 역변환 필요 (0~1 → 10~95)

    입력:
    - face_features: [batch, 6] - MediaPipe 얼굴 측정값
    - skin_features: [batch, 2] - MediaPipe 피부 측정값
    - style_emb: [batch, 384] - 헤어스타일 임베딩

    Note:
    - use_attention=True는 기존 체크포인트 호환성을 위해 유지
    - 단일 토큰 attention은 실질적으로 LayerNorm+Dropout과 동일 (identity에 가까움)
    - 새로운 학습 시에는 RecommendationModelV6 사용 권장
    """

    def __init__(
        self,
        face_feat_dim: int = 6,
        skin_feat_dim: int = 2,
        style_embed_dim: int = 384,
        use_attention: bool = True,  # 기존 체크포인트 호환성 유지
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.face_feat_dim = face_feat_dim
        self.skin_feat_dim = skin_feat_dim
        self.style_embed_dim = style_embed_dim

        # Input projection layers
        self.face_projection = nn.Sequential(
            nn.Linear(face_feat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        self.skin_projection = nn.Sequential(
            nn.Linear(skin_feat_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        self.total_dim = 64 + 32 + style_embed_dim  # 480

        # Attention layer (단일 토큰 - 실질적으로 LayerNorm+Dropout)
        # WARNING: 이 구조는 의미 없는 attention 적용임
        # 새로운 모델에서는 MultiTokenAttentionLayer 사용 권장
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionLayer(
                embed_dim=self.total_dim, num_heads=8, dropout=0.1
            )

        self.fc1 = nn.Linear(self.total_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate * 0.7)

        self.residual_proj = nn.Linear(self.total_dim, 128)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout_rate * 0.5)

        self.fc4 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)

        # Sigmoid 활성화 함수 - 출력을 0~1로 제한
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        face_features: torch.Tensor,
        skin_features: torch.Tensor,
        style_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass - 출력 범위: 0~1 (Sigmoid)"""
        face_proj = self.face_projection(face_features)
        skin_proj = self.skin_projection(skin_features)

        x = torch.cat([face_proj, skin_proj, style_emb], dim=1)

        # Apply attention if enabled (단일 토큰이므로 실질적으로 LayerNorm)
        if self.use_attention:
            x_att = x.unsqueeze(1)
            x_att = self.attention(x_att)
            x = x_att.squeeze(1)

        residual = self.residual_proj(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = x + residual

        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = torch.relu(x)

        x = self.fc_out(x)

        # Sigmoid로 0~1 출력 보장
        x = self.sigmoid(x)

        return x.squeeze(-1)


class RecommendationModelV6(nn.Module):
    """
    Multi-Token Attention 기반 추천 모델 v6

    v4/v5의 문제점 해결:
    - 기존: 단일 토큰(480차원)에 self-attention → 사실상 identity
    - 개선: 3개 토큰(face, skin, style) 간 cross-attention → 의미 있는 상호작용 학습

    구조:
    1. Input Projection: face(6→64), skin(2→32)
    2. Multi-Token Attention: 3개 토큰을 128차원으로 통일 후 attention
    3. Feature Fusion: attention 출력(384) → MLP → score

    입력:
    - face_features: [batch, 6] - MediaPipe 얼굴 측정값
    - skin_features: [batch, 2] - MediaPipe 피부 측정값
    - style_emb: [batch, 384] - 헤어스타일 임베딩

    출력:
    - normalized score: 0~1 (Sigmoid)
    """

    def __init__(
        self,
        face_feat_dim: int = 6,
        skin_feat_dim: int = 2,
        style_embed_dim: int = 384,
        token_dim: int = 128,
        num_heads: int = 4,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.face_feat_dim = face_feat_dim
        self.skin_feat_dim = skin_feat_dim
        self.style_embed_dim = style_embed_dim

        # Input projection layers (attention 이전)
        self.face_projection = nn.Sequential(
            nn.Linear(face_feat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        self.skin_projection = nn.Sequential(
            nn.Linear(skin_feat_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        # Multi-Token Attention Layer (3개 토큰 간 상호작용)
        self.multi_token_attention = MultiTokenAttentionLayer(
            face_dim=64,
            skin_dim=32,
            style_dim=style_embed_dim,
            token_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout_rate * 0.3,
        )

        # Attention 출력 차원: token_dim * 3 = 384
        attention_out_dim = token_dim * 3

        # Feature fusion network
        self.fc1 = nn.Linear(attention_out_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate * 0.7)

        # Residual connection
        self.residual_proj = nn.Linear(attention_out_dim, 128)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout_rate * 0.5)

        self.fc4 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)

        # Sigmoid 활성화 함수 - 출력을 0~1로 제한
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        face_features: torch.Tensor,
        skin_features: torch.Tensor,
        style_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass - 출력 범위: 0~1 (Sigmoid)"""
        # 1. Input projection
        face_proj = self.face_projection(face_features)  # (batch, 64)
        skin_proj = self.skin_projection(skin_features)  # (batch, 32)

        # 2. Multi-Token Attention (3개 토큰 간 상호작용)
        x = self.multi_token_attention(face_proj, skin_proj, style_emb)  # (batch, 384)

        # Store for residual
        residual = self.residual_proj(x)

        # 3. Feature fusion MLP
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        # Add residual connection
        x = x + residual

        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = torch.relu(x)

        x = self.fc_out(x)

        # Sigmoid로 0~1 출력 보장
        x = self.sigmoid(x)

        return x.squeeze(-1)


class MLHairstyleRecommender:
    """ML 기반 헤어스타일 추천기"""

    # MediaPipe와 호환되는 카테고리
    FACE_SHAPES = ["각진형", "둥근형", "긴형", "계란형"]
    SKIN_TONES = ["겨울쿨", "가을웜", "봄웜", "여름쿨"]

    def __init__(
        self,
        model_path: str = "models/hairstyle_recommender_v6_multitoken.pt",
        embeddings_path: str = "data_source/style_embeddings.npz",
        gender_metadata_path: str = "data_source/hairstyle_gender.json",
    ):
        """
        초기화

        Args:
            model_path: 학습된 모델 경로 (기본: v6 Multi-Token Attention)
            embeddings_path: 헤어스타일 임베딩 경로
            gender_metadata_path: 헤어스타일 성별 메타데이터 경로
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 모델 로드
        logger.info(f"📂 ML 모델 로딩: {model_path}")

        # 체크포인트 형식으로 저장된 경우 처리
        try:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )

            # 모델 버전 및 설정 확인
            if isinstance(checkpoint, dict) and "config" in checkpoint:
                config = checkpoint["config"]
                self.is_normalized_model = config.get("normalized", False)
                self.model_version = config.get(
                    "version", "v5" if self.is_normalized_model else "v4"
                )
                self.attention_type = config.get("attention_type", "single_token")
                logger.info(f"  - 모델 버전: {self.model_version}")
                logger.info(f"  - 정규화 모델: {self.is_normalized_model}")
                logger.info(f"  - Attention 타입: {self.attention_type}")
            else:
                self.is_normalized_model = False
                self.model_version = "v4"
                self.attention_type = "single_token"

            # 모델 클래스 선택
            if self.model_version == "v6" or self.attention_type == "multi_token":
                # V6: Multi-Token Attention
                token_dim = config.get("token_dim", 128)
                num_heads = config.get("num_heads", 4)
                self.model = RecommendationModelV6(
                    token_dim=token_dim, num_heads=num_heads
                )
                logger.info(
                    f"  - 사용 모델: RecommendationModelV6 (Multi-Token Attention)"
                )
            elif self.is_normalized_model:
                # V5: Normalized + Single-Token Attention
                self.model = NormalizedRecommendationModel()
                logger.info(
                    "  - 사용 모델: NormalizedRecommendationModel (v5 - Sigmoid 출력)"
                )
            else:
                # V4: Legacy
                self.model = RecommendationModel()
                logger.info("  - 사용 모델: RecommendationModel (v4)")

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(
                    f"✅ 체크포인트에서 모델 로드 완료 (epoch: {checkpoint.get('epoch', 'N/A')})"
                )
            else:
                self.model.load_state_dict(checkpoint)
                logger.info(f"✅ 모델 로드 완료")
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {str(e)}")
            raise

        self.model.to(self.device)
        self.model.eval()  # 추론 모드
        logger.info(f"✅ 모델 준비 완료 (디바이스: {self.device})")

        # 2. 헤어스타일 임베딩 로드
        logger.info(f"📂 임베딩 로딩: {embeddings_path}")
        try:
            data = np.load(embeddings_path, allow_pickle=False)
            self.styles = data["styles"].tolist()  # 헤어스타일명 리스트
            self.embeddings = data["embeddings"]  # (N, 384) 임베딩
            logger.info(f"✅ 임베딩 로드 완료: {len(self.styles)}개 스타일")
        except Exception as e:
            logger.error(f"❌ 임베딩 로드 실패: {str(e)}")
            raise

        # 스타일명 -> 인덱스 매핑
        self.style_to_idx = {style: idx for idx, style in enumerate(self.styles)}

        # 3. 성별 메타데이터 로드 (NEW)
        logger.info(f"📂 성별 메타데이터 로딩: {gender_metadata_path}")
        try:
            import json
            import os

            if os.path.exists(gender_metadata_path):
                with open(gender_metadata_path, "r", encoding="utf-8") as f:
                    self.gender_metadata = json.load(f)
                logger.info(
                    f"✅ 성별 메타데이터 로드 완료: {len(self.gender_metadata)}개 스타일"
                )
            else:
                logger.warning(f"⚠️ 성별 메타데이터 파일 없음 - 성별 필터링 비활성화")
                self.gender_metadata = {}
        except Exception as e:
            logger.error(f"❌ 성별 메타데이터 로드 실패: {str(e)}")
            self.gender_metadata = {}

        # 4. 실시간 임베딩용 SentenceTransformer 로드 (Lambda에서는 스킵)
        import os

        is_lambda = os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None

        if not is_lambda:
            logger.info(
                "🔄 실시간 임베딩 모델 로딩 (paraphrase-multilingual-MiniLM-L12-v2)..."
            )
            try:
                from sentence_transformers import SentenceTransformer

                self.sentence_model = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2"
                )
                logger.info("✅ 실시간 임베딩 모델 준비 완료")
            except Exception as e:
                logger.error(f"❌ 실시간 임베딩 모델 로드 실패: {str(e)}")
                self.sentence_model = None
        else:
            logger.info("🔧 Lambda 환경 - 실시간 임베딩 모델 스킵")
            self.sentence_model = None

    def _encode_face_shape(self, face_shape: str) -> np.ndarray:
        """얼굴형을 one-hot 인코딩 (6차원 - 모델과 일치)"""
        vec = np.zeros(6, dtype=np.float32)

        # 하트형은 계란형으로 매핑
        if face_shape == "하트형":
            face_shape = "계란형"
            logger.debug("하트형을 계란형으로 매핑")

        # 기본 4가지 얼굴형에 대한 one-hot 인코딩
        if face_shape in self.FACE_SHAPES:
            idx = self.FACE_SHAPES.index(face_shape)
            vec[idx] = 1.0
        else:
            logger.warning(f"알 수 없는 얼굴형: {face_shape}, 계란형으로 기본값 사용")
            vec[3] = 1.0  # 계란형

        # 추가 특징 차원 (모델 학습 시 사용됨)
        vec[4] = 0.5  # 중간 값으로 초기화
        vec[5] = 0.5  # 중간 값으로 초기화

        return vec

    def _encode_skin_tone(self, skin_tone: str) -> np.ndarray:
        """피부톤을 one-hot 인코딩 (2차원 - 모델과 일치)"""
        vec = np.zeros(2, dtype=np.float32)

        # 봄/가을 -> 웜톤(0), 여름/겨울 -> 쿨톤(1)
        if skin_tone in ["봄웜", "가을웜"]:
            vec[0] = 1.0  # 웜톤
        elif skin_tone in ["여름쿨", "겨울쿨"]:
            vec[1] = 1.0  # 쿨톤
        else:
            logger.warning(f"알 수 없는 피부톤: {skin_tone}, 웜톤으로 기본값 사용")
            vec[0] = 1.0  # 웜톤

        return vec

    # 스타일 카테고리 키워드 (같은 카테고리는 유사한 스타일로 간주)
    # 주의: 더 구체적인 키워드가 먼저 와야 함
    STYLE_CATEGORY_KEYWORDS = [
        ["쉼표머리", "쉼표 머리", "comma"],  # 쉼표머리 계열 (먼저 체크)
        ["히피머리", "히피", "hippie"],  # 히피 계열
        ["가르마", "센터", "사이드"],  # 가르마 계열
        ["시스루", "풀뱅"],  # 시스루/풀뱅 계열
        ["레이어드", "레이어"],  # 레이어드 계열
        ["숏컷", "숏헤어", "짧은", "short"],  # 숏컷 계열
        ["롱 스타일", "롱헤어", "긴머리", "장발", "long"],  # 롱 계열
        ["투블럭", "투블록", "언더컷"],  # 투블럭 계열
        ["댄디", "포마드", "슬릭백"],  # 댄디 계열
        ["보브", "단발", "bob"],  # 보브/단발 계열
        ["머쉬룸", "버섯"],  # 머쉬룸 계열
        ["펌", "웨이브", "컬", "perm"],  # 펌 계열
        ["심플", "자연", "내추럴"],  # 심플/자연 계열
    ]

    def _get_style_category(self, style_name: str) -> int:
        """스타일명에서 카테고리 추출 (0-based index, -1이면 카테고리 없음)"""
        style_lower = style_name.lower()
        for idx, keywords in enumerate(self.STYLE_CATEGORY_KEYWORDS):
            for keyword in keywords:
                if keyword in style_lower:
                    return idx
        return -1

    def _is_similar_style(
        self, style_a: str, style_b: str, threshold: float = 0.65
    ) -> bool:
        """
        두 스타일명의 유사도 계산 (문자열 유사도 + 카테고리 기반)

        Args:
            style_a: 첫 번째 스타일명
            style_b: 두 번째 스타일명
            threshold: 유사도 임계값 (기본 0.65 = 65%)

        Returns:
            threshold 이상이면 True (유사한 스타일)

        로직:
            1. 문자열 유사도가 threshold 이상이면 유사함
            2. 같은 카테고리 키워드를 포함하면 유사함 (예: "센터 가르마" vs "가르마 스타일")
        """
        # 1. 문자열 유사도 체크
        ratio = SequenceMatcher(None, style_a, style_b).ratio()
        if ratio >= threshold:
            return True

        # 2. 카테고리 기반 체크 (같은 카테고리면 유사함)
        cat_a = self._get_style_category(style_a)
        cat_b = self._get_style_category(style_b)

        if cat_a >= 0 and cat_a == cat_b:
            logger.debug(
                f"[DIVERSITY] 같은 카테고리: '{style_a}' vs '{style_b}' (category={cat_a})"
            )
            return True

        return False

    def _get_style_embedding(self, style_name: str) -> np.ndarray:
        """
        스타일 임베딩 가져오기 (DB 조회 또는 실시간 생성)

        Args:
            style_name: 헤어스타일명 (정규화된 이름 권장)

        Returns:
            임베딩 벡터 (384,) 또는 None
        """
        # 1. DB 조회 (Fast Path)
        if style_name in self.style_to_idx:
            idx = self.style_to_idx[style_name]
            return self.embeddings[idx]

        # 2. 실시간 생성 (Slow Path)
        if self.sentence_model:
            logger.info(f"🆕 새로운 스타일 발견: '{style_name}' -> 실시간 임베딩 생성")
            try:
                embedding = self.sentence_model.encode(style_name)
                return embedding
            except Exception as e:
                logger.error(f"❌ 임베딩 생성 실패 ({style_name}): {str(e)}")
                return None

        return None

    def predict_score(self, face_shape: str, skin_tone: str, hairstyle: str) -> float:
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

        # 임베딩 가져오기 (DB or 실시간)
        style_embedding = self._get_style_embedding(normalized_style)

        if style_embedding is None:
            # 원본 이름으로도 시도
            style_embedding = self._get_style_embedding(hairstyle)

            if style_embedding is None:
                logger.warning(f"임베딩 생성 불가: '{hairstyle}'")
                return 0.0

        # 개별 특징 벡터 생성
        face_vec = self._encode_face_shape(face_shape)  # (4,)
        tone_vec = self._encode_skin_tone(skin_tone)  # (4,)

        # 모델 추론 - 3개의 개별 텐서로 전달
        with torch.no_grad():
            face_tensor = torch.FloatTensor(face_vec).unsqueeze(0).to(self.device)
            skin_tensor = torch.FloatTensor(tone_vec).unsqueeze(0).to(self.device)
            style_tensor = (
                torch.FloatTensor(style_embedding).unsqueeze(0).to(self.device)
            )

            score_tensor = self.model(face_tensor, skin_tensor, style_tensor)
            score = score_tensor.cpu().item()

        # 정규화 모델(v5)인 경우 역변환 적용 (0~1 → 10~95)
        if self.is_normalized_model:
            score = denormalize_score(score)

        # 10-95 범위로 클리핑 (정규화 모델) 또는 0-100 (기존 모델)
        if self.is_normalized_model:
            score = max(LABEL_MIN, min(LABEL_MAX, score))
        else:
            score = max(0.0, min(100.0, score))

        return round(score, 2)

    def recommend_top_k(
        self,
        face_shape: str = None,
        skin_tone: str = None,
        k: int = 3,
        face_features: List[float] = None,
        skin_features: List[float] = None,
        gender: str = None,
    ) -> List[Dict[str, any]]:
        """
        Top-K 헤어스타일 추천 (성별 필터링 적용)

        Args:
            face_shape: 얼굴형 (예: "계란형") - DEPRECATED, 하위 호환성을 위해 유지
            skin_tone: 피부톤 (예: "봄웜") - DEPRECATED, 하위 호환성을 위해 유지
            k: 추천 개수
            face_features: MediaPipe 얼굴 측정값 [face_ratio, forehead_width, cheekbone_width, jaw_width, forehead_ratio, jaw_ratio] (6차원)
            skin_features: MediaPipe 피부 측정값 [ITA_value, hue_value] (2차원)
            gender: 성별 ("male", "female", "neutral") - MediaPipe로 추론된 값

        Returns:
            추천 리스트 [{"hairstyle": "...", "score": 85.3}, ...]
        """
        # 실제 측정값 우선 사용, 없으면 라벨 기반 인코딩 (하위 호환성)
        if face_features is not None and skin_features is not None:
            logger.info(f"[ML DEBUG] ML 추천 시작 (실제 측정값 사용) - Top-{k}")
            logger.info(f"[ML DEBUG] Face features (원본): {face_features}")
            logger.info(f"[ML DEBUG] Skin features (원본): {skin_features}")

            # NumPy 배열로 변환
            face_vec = np.array(face_features, dtype=np.float32)
            tone_vec = np.array(skin_features, dtype=np.float32)

            # 차원 검증
            if face_vec.shape[0] != 6:
                raise ValueError(
                    f"face_features는 6차원이어야 합니다. 현재: {face_vec.shape[0]}"
                )
            if tone_vec.shape[0] != 2:
                raise ValueError(
                    f"skin_features는 2차원이어야 합니다. 현재: {tone_vec.shape[0]}"
                )

            # 입력 스케일링 적용 (학습 데이터 분포에 맞게 변환)
            face_vec, tone_vec = scale_input_features(face_vec, tone_vec)
            logger.info(f"[ML DEBUG] Face features (스케일링 후): {face_vec.tolist()}")
            logger.info(f"[ML DEBUG] Skin features (스케일링 후): {tone_vec.tolist()}")
        else:
            # 하위 호환성: 라벨 기반 인코딩
            logger.warning(
                f"[ML DEPRECATED] 라벨 기반 인코딩 사용: {face_shape} + {skin_tone}"
            )
            logger.warning(
                "[ML DEPRECATED] 실제 측정값(face_features, skin_features)을 전달하는 것을 권장합니다."
            )

            if face_shape is None or skin_tone is None:
                raise ValueError(
                    "face_features와 skin_features가 없으면 face_shape과 skin_tone을 제공해야 합니다."
                )

            face_vec = self._encode_face_shape(face_shape)  # (6,)
            tone_vec = self._encode_skin_tone(skin_tone)  # (2,)

        logger.info(f"[ML DEBUG] Face vector: {face_vec.tolist()}")
        logger.info(f"[ML DEBUG] Skin vector: {tone_vec.tolist()}")

        # 모든 헤어스타일에 대해 점수 예측
        all_scores = []

        # 배치 처리로 최적화
        batch_size = 64
        num_styles = len(self.styles)

        for i in range(0, num_styles, batch_size):
            batch_end = min(i + batch_size, num_styles)
            batch_size_actual = batch_end - i
            batch_embeddings = self.embeddings[i:batch_end]

            # 배치 추론 - 3개의 개별 텐서로 전달
            with torch.no_grad():
                # 얼굴형과 피부톤은 배치 크기만큼 복제
                face_batch = np.tile(face_vec, (batch_size_actual, 1))
                skin_batch = np.tile(tone_vec, (batch_size_actual, 1))

                face_tensor = torch.FloatTensor(face_batch).to(self.device)
                skin_tensor = torch.FloatTensor(skin_batch).to(self.device)
                style_tensor = torch.FloatTensor(batch_embeddings).to(self.device)

                # 첫 번째 배치에서만 디버그 정보 출력
                if i == 0:
                    logger.info(
                        f"[ML DEBUG] First batch embedding shape: {batch_embeddings.shape}"
                    )
                    logger.info(
                        f"[ML DEBUG] First style embedding std: {batch_embeddings.std():.6f}"
                    )
                    logger.info(f"[ML DEBUG] First 3 styles: {self.styles[i:i+3]}")

                scores_tensor = self.model(face_tensor, skin_tensor, style_tensor)
                scores = scores_tensor.cpu().numpy().flatten()

                # 정규화 모델(v5)인 경우 역변환 적용 (0~1 → 10~95)
                if self.is_normalized_model:
                    scores = scores * LABEL_RANGE + LABEL_MIN

                # 첫 번째 배치에서만 점수 디버그
                if i == 0:
                    logger.info(f"[ML DEBUG] First batch scores: {scores[:5].tolist()}")
                    logger.info(f"[ML DEBUG] Scores std: {scores.std():.6f}")
                    if self.is_normalized_model:
                        logger.info(
                            f"[ML DEBUG] 정규화 모델 - 역변환 적용됨 (0~1 → {LABEL_MIN}~{LABEL_MAX})"
                        )

            # 결과 저장
            for j, score in enumerate(scores):
                style_idx = i + j
                all_scores.append(
                    {
                        "hairstyle_id": style_idx,  # DB ID 추가
                        "hairstyle": self.styles[style_idx],
                        "score": float(score),  # 역변환된 점수 (10~95 범위)
                        "original_score": float(score),  # 피드백용 원본 점수 보존
                    }
                )

        # 점수 기준 정렬
        all_scores.sort(key=lambda x: x["score"], reverse=True)

        # 성별 필터링 (NEW)
        if gender and self.gender_metadata:
            logger.info(
                f"[GENDER] 성별 필터링 시작 (gender={gender}, metadata_count={len(self.gender_metadata)})"
            )
            filtered_scores = []
            debug_count = 0
            for item in all_scores:
                style_name = item["hairstyle"]
                style_gender = self.gender_metadata.get(style_name, "unisex")

                # 처음 5개 스타일은 디버깅용 로깅
                if debug_count < 5:
                    logger.debug(
                        f"[GENDER DEBUG] {style_name}: {style_gender} (score={item['score']:.2f})"
                    )
                    debug_count += 1

                # 성별 매칭 로직:
                # - neutral (애매한 경우): 모든 스타일 추천
                # - male: male + unisex 추천
                # - female: female + unisex 추천
                if gender == "neutral":
                    filtered_scores.append(item)
                elif gender == "male" and style_gender in ["male", "unisex"]:
                    filtered_scores.append(item)
                elif gender == "female" and style_gender in ["female", "unisex"]:
                    filtered_scores.append(item)

            logger.info(
                f"[GENDER] 필터링 완료: {len(all_scores)}개 → {len(filtered_scores)}개 "
                f"(제외: {len(all_scores) - len(filtered_scores)}개)"
            )
            all_scores = filtered_scores
        else:
            if not gender:
                logger.warning(
                    "[GENDER] 성별 필터링 비활성화 - gender 파라미터가 비어있음!"
                )
            elif not self.gender_metadata:
                logger.warning(
                    "[GENDER] 성별 필터링 비활성화 - metadata가 로드되지 않음!"
                )
            else:
                logger.info("[GENDER] 성별 필터링 비활성화")

        # 유사도 기반 다양성 필터링 (65% 이상 유사한 스타일 제외)
        top_k_recommendations = []
        similarity_threshold = 0.65
        max_candidates = min(100, len(all_scores))  # 상위 100개까지 탐색

        logger.info(
            f"[DIVERSITY] 다양성 필터링 시작 (threshold={similarity_threshold})"
        )

        for candidate in all_scores[:max_candidates]:
            if len(top_k_recommendations) >= k:
                break

            candidate_style = candidate["hairstyle"]

            # 이미 선택된 스타일과 유사도 체크
            is_duplicate = False
            for selected in top_k_recommendations:
                selected_style = selected["hairstyle"]
                if self._is_similar_style(
                    candidate_style, selected_style, similarity_threshold
                ):
                    logger.debug(
                        f"[DIVERSITY] 유사한 스타일 제외: '{candidate_style}' "
                        f"(유사: '{selected_style}')"
                    )
                    is_duplicate = True
                    break

            if not is_duplicate:
                top_k_recommendations.append(candidate)
                logger.info(
                    f"[DIVERSITY] 선택 ({len(top_k_recommendations)}/{k}): "
                    f"'{candidate_style}' (점수: {candidate['score']:.2f})"
                )

        # k개를 채우지 못한 경우 경고
        if len(top_k_recommendations) < k:
            logger.warning(
                f"[DIVERSITY] 다양한 스타일 {k}개를 찾지 못함 "
                f"(실제: {len(top_k_recommendations)}개). "
                f"threshold를 낮추거나 데이터를 확인하세요."
            )

        # 점수 스케일링 처리
        # - v5 정규화 모델: 이미 10~95 범위이므로 원본 점수 그대로 사용
        # - v4 기존 모델: Min-Max 정규화로 75~95 범위로 스케일링
        if self.is_normalized_model:
            # v5 정규화 모델: 원본 점수 그대로 사용 (이미 10~95 범위)
            logger.info(f"[SCORE] v5 정규화 모델 - 원본 점수 사용")
            for rec in top_k_recommendations:
                rec["score"] = round(rec["original_score"], 2)
        elif len(top_k_recommendations) >= 2:
            # v4 기존 모델: Min-Max 정규화를 사용한 점수 스케일링
            # Top-K 내에서 점수를 75~95점 범위로 정규화
            raw_scores = [rec["original_score"] for rec in top_k_recommendations]
            min_raw = min(raw_scores)
            max_raw = max(raw_scores)

            if max_raw > min_raw:
                target_min, target_max = 75.0, 95.0

                logger.info(f"[SCORE NORM] Raw scores: {raw_scores}")
                logger.info(f"[SCORE NORM] Raw range: {min_raw:.2f} ~ {max_raw:.2f}")

                for rec in top_k_recommendations:
                    raw = rec["original_score"]
                    normalized = (raw - min_raw) / (max_raw - min_raw) * (
                        target_max - target_min
                    ) + target_min
                    rec["score"] = round(normalized, 2)

                logger.info(
                    f"[SCORE NORM] Normalized scores: {[r['score'] for r in top_k_recommendations]}"
                )
            else:
                for i, rec in enumerate(top_k_recommendations):
                    rec["score"] = round(95.0 - i * 3, 2)
                logger.info(
                    f"[SCORE NORM] Same scores - using fallback: {[r['score'] for r in top_k_recommendations]}"
                )
        elif len(top_k_recommendations) == 1:
            top_k_recommendations[0]["score"] = 90.0
            logger.info("[SCORE NORM] Single recommendation - set to 90.0")

        # 디버그: Top-K 점수 분포
        if top_k_recommendations:
            scores_list = [r["score"] for r in top_k_recommendations]
            logger.info(f"[ML DEBUG] Top-{k} final scores: {scores_list}")
            logger.info(
                f"[ML DEBUG] Score range: {min(scores_list):.2f} ~ {max(scores_list):.2f}"
            )

        logger.info(
            f"[ML RESULT] ML 추천 완료: {[r['hairstyle'] for r in top_k_recommendations]}"
        )

        return top_k_recommendations

    def batch_predict(
        self, face_shape: str, skin_tone: str, hairstyles: List[str]
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

        # 1. 임베딩 수집 (DB or 실시간)
        valid_styles = []
        batch_embeddings = []

        for style in hairstyles:
            normalized = normalize_style_name(style)
            embedding = self._get_style_embedding(normalized)

            if embedding is None:
                # 원본 이름으로도 시도
                embedding = self._get_style_embedding(style)

            if embedding is not None:
                valid_styles.append(style)
                batch_embeddings.append(embedding)
            else:
                logger.warning(f"임베딩 생성 불가로 건너뜀: {style}")

        if not valid_styles:
            logger.warning("유효한 헤어스타일이 없습니다")
            return results

        # 2. 얼굴형과 피부톤 특징 벡터 생성
        face_vec = self._encode_face_shape(face_shape)  # (4,)
        tone_vec = self._encode_skin_tone(skin_tone)  # (4,)

        batch_embeddings = np.array(batch_embeddings, dtype=np.float32)

        # 3. 배치 추론 - 3개의 개별 텐서로 전달
        with torch.no_grad():
            batch_size = len(valid_styles)
            face_batch = np.tile(face_vec, (batch_size, 1))
            skin_batch = np.tile(tone_vec, (batch_size, 1))

            face_tensor = torch.FloatTensor(face_batch).to(self.device)
            skin_tensor = torch.FloatTensor(skin_batch).to(self.device)
            style_tensor = torch.FloatTensor(batch_embeddings).to(self.device)

            scores_tensor = self.model(face_tensor, skin_tensor, style_tensor)
            scores = scores_tensor.cpu().numpy().flatten()

        # 정규화 모델(v5)인 경우 역변환 적용 (0~1 → 10~95)
        if self.is_normalized_model:
            scores = scores * LABEL_RANGE + LABEL_MIN

        # 4. 결과 저장
        for style, score in zip(valid_styles, scores):
            if self.is_normalized_model:
                score_clipped = max(LABEL_MIN, min(LABEL_MAX, float(score)))
            else:
                score_clipped = max(0.0, min(100.0, float(score)))
            results[style] = round(score_clipped, 2)

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


# ========== A/B 테스트 래퍼 클래스 ==========


class ABTestRecommender:
    """
    A/B 테스트를 지원하는 ML 추천기 래퍼

    Champion(기존 모델)과 Challenger(신규 모델)을 관리하고,
    사용자별로 일관된 모델을 선택하여 추천을 수행합니다.

    특징:
    - Lazy Loading: 모델은 처음 사용될 때 로드됨
    - 메모리 최적화: Challenger 비활성화 시 Champion만 로드
    - 일관된 라우팅: 동일 user_id는 항상 동일 모델 사용

    Usage:
        recommender = get_ab_recommender()
        results = recommender.recommend_top_k(
            user_id="analysis_123",
            face_features=[...],
            skin_features=[...],
            gender="male",
            k=3
        )
        # results에는 model_version, experiment_id, ab_variant 포함
    """

    def __init__(self):
        """초기화 - 모델은 Lazy Loading"""
        self._champion_model: Optional[MLHairstyleRecommender] = None
        self._challenger_model: Optional[MLHairstyleRecommender] = None
        self._router = None

        # A/B 테스트 라우터 초기화
        self._init_router()

    def _init_router(self):
        """A/B 테스트 라우터 초기화"""
        try:
            from services.mlops.ab_test import get_ab_router

            self._router = get_ab_router()
            logger.info(
                f"🔬 A/B 테스트 라우터 초기화 완료 (enabled={self._router.is_abtest_active()})"
            )
        except ImportError:
            logger.warning("⚠️ A/B 테스트 모듈 로드 실패 - Champion 모델만 사용")
            self._router = None

    @property
    def champion_model(self) -> MLHairstyleRecommender:
        """Champion 모델 (Lazy Loading)"""
        if self._champion_model is None:
            logger.info("🔧 Champion 모델 로딩...")
            # 기본 경로에서 로드 (기존 싱글톤과 동일)
            self._champion_model = get_ml_recommender()
        return self._champion_model

    @property
    def challenger_model(self) -> Optional[MLHairstyleRecommender]:
        """Challenger 모델 (Lazy Loading, A/B 테스트 활성화 시에만)"""
        if (
            self._challenger_model is None
            and self._router
            and self._router.is_abtest_active()
        ):
            logger.info("🔧 Challenger 모델 로딩...")
            try:
                # Challenger 모델 경로 결정
                challenger_version = self._router.config.challenger_model_version
                challenger_path = f"models/challenger/model.pt"

                # S3에서 다운로드 필요시 여기서 처리
                # 현재는 로컬 경로 사용
                import os

                if os.path.exists(challenger_path):
                    self._challenger_model = MLHairstyleRecommender(
                        model_path=challenger_path
                    )
                    logger.info(f"✅ Challenger 모델 로드 완료: {challenger_version}")
                else:
                    logger.warning(f"⚠️ Challenger 모델 파일 없음: {challenger_path}")
            except Exception as e:
                logger.error(f"❌ Challenger 모델 로드 실패: {e}")

        return self._challenger_model

    def recommend_top_k(
        self,
        user_id: str,
        face_features: List[float] = None,
        skin_features: List[float] = None,
        gender: str = None,
        k: int = 3,
        face_shape: str = None,
        skin_tone: str = None,
    ) -> List[Dict[str, any]]:
        """
        Top-K 헤어스타일 추천 (A/B 테스트 적용)

        Args:
            user_id: 사용자 ID (analysis_id 사용 가능) - 일관된 라우팅용
            face_features: MediaPipe 얼굴 측정값 (6차원)
            skin_features: MediaPipe 피부 측정값 (2차원)
            gender: 성별 ("male", "female", "neutral")
            k: 추천 개수
            face_shape: 얼굴형 (deprecated, 하위 호환성)
            skin_tone: 피부톤 (deprecated, 하위 호환성)

        Returns:
            추천 리스트 [{"hairstyle": "...", "score": 85.3, "model_version": "v6", "experiment_id": "...", "ab_variant": "champion"}, ...]
        """
        # 변형 결정
        variant = None
        experiment_info = {}

        if self._router:
            from services.mlops.ab_test import ModelVariant

            variant = self._router.get_variant(user_id)
            experiment_info = self._router.get_experiment_info(variant)
        else:
            experiment_info = {
                "experiment_id": "",
                "model_version": "v6",
                "ab_variant": "champion",
            }

        # 모델 선택
        model = self.champion_model  # 기본값

        if variant and self._router:
            from services.mlops.ab_test import ModelVariant

            if variant == ModelVariant.CHALLENGER and self.challenger_model:
                model = self.challenger_model
                logger.debug(f"[ABTEST] user={user_id} -> Challenger 모델 사용")
            else:
                logger.debug(f"[ABTEST] user={user_id} -> Champion 모델 사용")

        # 추천 수행
        results = model.recommend_top_k(
            face_shape=face_shape,
            skin_tone=skin_tone,
            k=k,
            face_features=face_features,
            skin_features=skin_features,
            gender=gender,
        )

        # 모델 버전 정보 추가 (피드백 분석용)
        for result in results:
            result["model_version"] = experiment_info.get("model_version", "v6")
            result["experiment_id"] = experiment_info.get("experiment_id", "")
            result["ab_variant"] = experiment_info.get("ab_variant", "champion")

        return results

    def is_abtest_active(self) -> bool:
        """A/B 테스트 활성화 여부"""
        return self._router is not None and self._router.is_abtest_active()

    def get_current_config(self) -> Dict[str, any]:
        """현재 A/B 테스트 설정 반환"""
        if self._router:
            return self._router.config.to_dict()
        return {"enabled": False}


# ========== A/B 테스트 추천기 싱글톤 ==========
_ab_recommender_instance: Optional[ABTestRecommender] = None


def get_ab_recommender() -> ABTestRecommender:
    """
    A/B 테스트 추천기 싱글톤 인스턴스 가져오기

    A/B 테스트가 비활성화되어 있어도 동작합니다 (Champion만 사용).

    Returns:
        ABTestRecommender 인스턴스
    """
    global _ab_recommender_instance

    if _ab_recommender_instance is None:
        logger.info("🔧 A/B 테스트 추천기 초기화 중...")
        _ab_recommender_instance = ABTestRecommender()
        logger.info("✅ A/B 테스트 추천기 준비 완료")

    return _ab_recommender_instance


# ========== 유틸리티 함수 (analyze.py 호환) ==========

# Confidence threshold 상수
CONFIDENCE_THRESHOLD_VERY_HIGH = 0.90
CONFIDENCE_THRESHOLD_HIGH = 0.85
CONFIDENCE_THRESHOLD_MEDIUM = 0.75


def predict_ml_score(face_shape: str, skin_tone: str, hairstyle: str) -> float:
    """
    ML 모델로 특정 헤어스타일의 추천 점수 예측

    Args:
        face_shape: 얼굴형 (예: "계란형")
        skin_tone: 피부톤 (예: "봄웜")
        hairstyle: 헤어스타일명 (예: "시스루뱅 단발")

    Returns:
        추천 점수 (0.0 ~ 100.0)
    """
    try:
        recommender = get_ml_recommender()
        score = recommender.predict_score(face_shape, skin_tone, hairstyle)
        return score
    except Exception as e:
        logger.error(f"ML 예측 실패: {str(e)}")
        return 85.0  # 기본값


def get_confidence_level(score: float) -> str:
    """
    점수를 신뢰도 레벨 문자열로 변환

    Args:
        score: 신뢰도 점수 (0.0 ~ 1.0 또는 0.0 ~ 100.0)

    Returns:
        신뢰도 레벨 문자열 ("매우 높음", "높음", "보통", "낮음")
    """
    # 0~100 범위 점수를 0~1로 정규화
    if score > 1.0:
        score = score / 100.0

    if score >= CONFIDENCE_THRESHOLD_VERY_HIGH:
        return "매우 높음"
    elif score >= CONFIDENCE_THRESHOLD_HIGH:
        return "높음"
    elif score >= CONFIDENCE_THRESHOLD_MEDIUM:
        return "보통"
    else:
        return "낮음"
