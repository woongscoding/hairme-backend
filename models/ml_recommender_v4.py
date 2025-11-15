#!/usr/bin/env python3
"""
ML 기반 헤어스타일 추천기 v4 - Continuous Features Architecture

✨ v4.0 개선사항:
- 연속형 변수 직접 입력 (MediaPipe 측정값)
- 범주형 임베딩 제거 (더 정확한 측정값 사용)
- Face features (6차원) + Skin features (2차원)
- Attention mechanism
- Residual connections
- 헤어스타일 임베딩 통합

Author: HairMe ML Team
Date: 2025-11-15
Version: 4.0.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """Multi-head self-attention layer"""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x)

        # Residual connection + layer norm
        x = self.norm(x + self.dropout(attn_out))

        return x


class ContinuousRecommenderV4(nn.Module):
    """
    연속형 변수 기반 추천 모델 v4

    입력:
    - face_features: [batch, 6] - MediaPipe 얼굴 측정값
      [face_ratio, forehead_width, cheekbone_width, jaw_width, forehead_ratio, jaw_ratio]
    - skin_features: [batch, 2] - MediaPipe 피부 측정값
      [ITA_value, hue_value]
    - style_emb: [batch, 384] - 헤어스타일 임베딩

    출력:
    - score: [batch, 1] - 추천 점수 (0-100)
    """

    def __init__(
        self,
        face_feat_dim: int = 6,
        skin_feat_dim: int = 2,
        style_embed_dim: int = 384,
        use_attention: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()

        self.face_feat_dim = face_feat_dim
        self.skin_feat_dim = skin_feat_dim
        self.style_embed_dim = style_embed_dim

        # Input projection layers (연속형 변수 → 고차원 임베딩)
        self.face_projection = nn.Sequential(
            nn.Linear(face_feat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )

        self.skin_projection = nn.Sequential(
            nn.Linear(skin_feat_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )

        # Total dimension after projection
        self.total_dim = 64 + 32 + style_embed_dim  # 96 + 384 = 480

        # Attention layer (optional)
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionLayer(
                embed_dim=self.total_dim,
                num_heads=8,
                dropout=0.1
            )

        # Feature fusion network with residual connections
        self.fc1 = nn.Linear(self.total_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate * 0.7)

        # Residual connection for dimension matching
        self.residual_proj = nn.Linear(self.total_dim, 128)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout_rate * 0.5)

        self.fc4 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        face_features: torch.Tensor,
        skin_features: torch.Tensor,
        style_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            face_features: [batch, 6] - MediaPipe 얼굴 측정값
            skin_features: [batch, 2] - MediaPipe 피부 측정값
            style_emb: [batch, 384] - 헤어스타일 임베딩

        Returns:
            Recommendation scores [batch, 1]
        """
        # Project continuous features to higher dimensions
        face_proj = self.face_projection(face_features)  # [batch, 64]
        skin_proj = self.skin_projection(skin_features)  # [batch, 32]

        # Concatenate all features
        combined = torch.cat([face_proj, skin_proj, style_emb], dim=-1)  # [batch, 480]

        # Apply attention if enabled
        if self.use_attention:
            # Reshape for attention (needs 3D input)
            combined_3d = combined.unsqueeze(1)  # [batch, 1, 480]
            attended = self.attention(combined_3d)  # [batch, 1, 480]
            combined = attended.squeeze(1)  # [batch, 480]

        # Store original for residual
        residual = self.residual_proj(combined)

        # First block
        x = self.fc1(combined)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Second block with residual
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x + residual  # Residual connection
        x = self.dropout2(x)

        # Third block
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        # Output layers
        x = self.fc4(x)
        x = F.relu(x)
        score = self.fc_out(x)

        return score


class MLHairstyleRecommenderV4:
    """
    ML 기반 헤어스타일 추천기 v4
    연속형 변수 기반 아키텍처
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_attention: bool = True
    ):
        """
        초기화

        Args:
            model_path: 학습된 모델 경로 (None이면 새 모델 생성)
            use_attention: Attention 사용 여부
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 모델 생성
        self.model = ContinuousRecommenderV4(
            face_feat_dim=6,
            skin_feat_dim=2,
            style_embed_dim=384,
            use_attention=use_attention
        )

        # 학습된 모델 로드 (있는 경우)
        if model_path and Path(model_path).exists():
            logger.info(f"📂 모델 로딩: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"  Epoch {checkpoint.get('epoch', 'unknown')}, "
                          f"Best Val Loss: {checkpoint.get('best_val_loss', 'unknown'):.3f}")
            else:
                self.model.load_state_dict(checkpoint)

            logger.info(f"✅ 모델 로드 완료 (디바이스: {self.device})")

        self.model.to(self.device)
        self.model.eval()

        # Sentence Transformer (헤어스타일 임베딩용)
        self.sentence_model = None
        try:
            from sentence_transformers import SentenceTransformer
            model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
            logger.info(f"📂 Sentence Transformer 로딩: {model_name}")
            self.sentence_model = SentenceTransformer(model_name)
            logger.info("✅ 실시간 임베딩 활성화")
        except Exception as e:
            logger.warning(f"⚠️ Sentence Transformer 로드 실패: {str(e)}")

    def predict_score(
        self,
        face_features: np.ndarray,
        skin_features: np.ndarray,
        hairstyle: str
    ) -> Tuple[float, Dict]:
        """
        헤어스타일 점수 예측

        Args:
            face_features: [6] - MediaPipe 얼굴 측정값
            skin_features: [2] - MediaPipe 피부 측정값
            hairstyle: 헤어스타일명

        Returns:
            (점수, 디버그 정보)
        """
        # 헤어스타일 임베딩 생성
        if self.sentence_model:
            style_emb = self.sentence_model.encode(hairstyle)
        else:
            # Fallback: 랜덤 임베딩 (실제로는 사용하지 말 것)
            style_emb = np.random.randn(384).astype(np.float32)
            logger.warning(f"⚠️ Sentence Transformer 없음, 랜덤 임베딩 사용")

        # 텐서 변환
        face_tensor = torch.tensor([face_features], dtype=torch.float32).to(self.device)
        skin_tensor = torch.tensor([skin_features], dtype=torch.float32).to(self.device)
        style_tensor = torch.tensor([style_emb], dtype=torch.float32).to(self.device)

        # 예측
        with torch.no_grad():
            score_tensor = self.model(face_tensor, skin_tensor, style_tensor)
            score = score_tensor.item()

        # 점수 범위 제한 (0-100)
        score = max(0.0, min(100.0, score))

        # 디버그 정보
        debug_info = {
            'face_features': face_features.tolist(),
            'skin_features': skin_features.tolist(),
            'raw_score': score_tensor.item(),
            'final_score': score,
            'hairstyle': hairstyle
        }

        return score, debug_info

    def predict_batch(
        self,
        face_features: np.ndarray,
        skin_features: np.ndarray,
        hairstyles: List[str]
    ) -> np.ndarray:
        """
        배치 예측

        Args:
            face_features: [batch, 6] - MediaPipe 얼굴 측정값
            skin_features: [batch, 2] - MediaPipe 피부 측정값
            hairstyles: 헤어스타일 리스트

        Returns:
            점수 배열 [batch]
        """
        # 헤어스타일 임베딩
        if self.sentence_model:
            style_embeddings = self.sentence_model.encode(hairstyles)
        else:
            style_embeddings = np.random.randn(len(hairstyles), 384).astype(np.float32)
            logger.warning(f"⚠️ Sentence Transformer 없음, 랜덤 임베딩 사용")

        # 텐서 변환
        face_tensor = torch.tensor(face_features, dtype=torch.float32).to(self.device)
        skin_tensor = torch.tensor(skin_features, dtype=torch.float32).to(self.device)
        style_tensor = torch.tensor(style_embeddings, dtype=torch.float32).to(self.device)

        # 예측
        with torch.no_grad():
            scores = self.model(face_tensor, skin_tensor, style_tensor)
            scores = scores.cpu().numpy().flatten()

        # 범위 제한
        scores = np.clip(scores, 0, 100)

        return scores


if __name__ == "__main__":
    # 테스트
    print("🧪 v4 모델 테스트")

    # 샘플 데이터
    face_feat = np.array([1.2, 200, 230, 180, 0.87, 0.78], dtype=np.float32)
    skin_feat = np.array([45.5, 25.0], dtype=np.float32)

    recommender = MLHairstyleRecommenderV4()

    # 단일 예측 테스트
    score, info = recommender.predict_score(face_feat, skin_feat, "댄디 컷")
    print(f"Score: {score:.1f}")
    print(f"Debug: {info}")

    # 배치 예측 테스트
    face_batch = np.array([face_feat, face_feat, face_feat])
    skin_batch = np.array([skin_feat, skin_feat, skin_feat])
    styles = ["댄디 컷", "투 블럭 컷", "리젠트 컷"]

    scores = recommender.predict_batch(face_batch, skin_batch, styles)
    print(f"Batch scores: {scores}")
