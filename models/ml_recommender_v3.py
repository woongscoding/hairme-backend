#!/usr/bin/env python3
"""
ML 기반 헤어스타일 추천기 v3 - Hybrid Architecture

✨ v3.0 개선사항:
- 학습가능한 Face/Tone 임베딩
- Attention mechanism
- Residual connections
- Combination-level evaluation 지원
- 새로운 조합에 대한 추론 능력

Author: HairMe ML Team
Date: 2025-11-11
Version: 3.0.0
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


class HybridRecommenderV3(nn.Module):
    """
    Hybrid 추천 모델 v3
    학습가능한 임베딩 + Attention + Residual connections
    """

    def __init__(
        self,
        n_face_shapes: int = 5,  # 사각형 포함
        n_skin_tones: int = 4,
        face_embed_dim: int = 32,
        tone_embed_dim: int = 32,
        style_embed_dim: int = 384,
        use_attention: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()

        # 학습가능한 임베딩 레이어
        self.face_embedding = nn.Embedding(n_face_shapes, face_embed_dim)
        self.tone_embedding = nn.Embedding(n_skin_tones, tone_embed_dim)

        # 임베딩 초기화 (Xavier)
        nn.init.xavier_uniform_(self.face_embedding.weight)
        nn.init.xavier_uniform_(self.tone_embedding.weight)

        # Total dimension
        self.total_dim = face_embed_dim + tone_embed_dim + style_embed_dim

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
        face_idx: torch.Tensor,
        tone_idx: torch.Tensor,
        style_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            face_idx: Face shape indices [batch_size]
            tone_idx: Skin tone indices [batch_size]
            style_emb: Style embeddings [batch_size, 384]

        Returns:
            Recommendation scores [batch_size, 1]
        """
        # Get embeddings
        face_emb = self.face_embedding(face_idx)  # [batch, face_dim]
        tone_emb = self.tone_embedding(tone_idx)  # [batch, tone_dim]

        # Concatenate all features
        combined = torch.cat([face_emb, tone_emb, style_emb], dim=-1)  # [batch, total_dim]

        # Apply attention if enabled
        if self.use_attention:
            # Reshape for attention (needs 3D input)
            combined_3d = combined.unsqueeze(1)  # [batch, 1, total_dim]
            attended = self.attention(combined_3d)  # [batch, 1, total_dim]
            combined = attended.squeeze(1)  # [batch, total_dim]

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


class MLHairstyleRecommenderV3:
    """
    ML 기반 헤어스타일 추천기 v3
    Hybrid architecture with learned embeddings
    """

    # 카테고리 정의
    FACE_SHAPES = ["각진형", "둥근형", "긴형", "계란형", "사각형"]
    SKIN_TONES = ["겨울쿨", "가을웜", "봄웜", "여름쿨"]

    def __init__(
        self,
        model_path: Optional[str] = None,
        embeddings_path: str = "data_source/style_embeddings.npz",
        use_attention: bool = True
    ):
        """
        초기화

        Args:
            model_path: 학습된 모델 경로 (None이면 새 모델 생성)
            embeddings_path: 헤어스타일 임베딩 경로
            use_attention: Attention 사용 여부
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Face/Tone 인덱스 매핑
        self.face_to_idx = {face: idx for idx, face in enumerate(self.FACE_SHAPES)}
        self.tone_to_idx = {tone: idx for idx, tone in enumerate(self.SKIN_TONES)}

        # 모델 생성
        self.model = HybridRecommenderV3(
            n_face_shapes=len(self.FACE_SHAPES),
            n_skin_tones=len(self.SKIN_TONES),
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

        # 헤어스타일 임베딩 로드
        logger.info(f"📂 임베딩 로딩: {embeddings_path}")
        data = np.load(embeddings_path, allow_pickle=True)
        self.styles = data['styles'].tolist()
        self.embeddings = data['embeddings']
        self.style_to_idx = {style: idx for idx, style in enumerate(self.styles)}
        logger.info(f"✅ 임베딩 로드 완료: {len(self.styles)}개 스타일")

        # Sentence Transformer (실시간 임베딩용)
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
        face_shape: str,
        skin_tone: str,
        hairstyle: str
    ) -> Tuple[float, Dict]:
        """
        헤어스타일 점수 예측

        Args:
            face_shape: 얼굴형
            skin_tone: 피부톤
            hairstyle: 헤어스타일명

        Returns:
            (점수, 디버그 정보)
        """
        # Face/Tone 인덱스 변환
        face_idx = self.face_to_idx.get(face_shape, 3)  # 기본값: 계란형
        tone_idx = self.tone_to_idx.get(skin_tone, 2)  # 기본값: 봄웜

        # 스타일 임베딩 가져오기
        from utils.style_preprocessor import normalize_style_name
        normalized = normalize_style_name(hairstyle)

        if normalized in self.style_to_idx:
            idx = self.style_to_idx[normalized]
            style_emb = self.embeddings[idx]
        elif self.sentence_model:
            # 실시간 임베딩 생성
            style_emb = self.sentence_model.encode(hairstyle)
            logger.info(f"✨ 실시간 임베딩 생성: '{hairstyle}'")
        else:
            # Fallback: 평균 임베딩
            style_emb = np.mean(self.embeddings, axis=0)
            logger.warning(f"⚠️ 미등록 스타일, 평균 임베딩 사용: '{hairstyle}'")

        # 텐서 변환
        face_tensor = torch.tensor([face_idx], dtype=torch.long).to(self.device)
        tone_tensor = torch.tensor([tone_idx], dtype=torch.long).to(self.device)
        style_tensor = torch.tensor([style_emb], dtype=torch.float32).to(self.device)

        # 예측
        with torch.no_grad():
            score_tensor = self.model(face_tensor, tone_tensor, style_tensor)
            score = score_tensor.item()

        # 점수 범위 제한 (0-100)
        score = max(0.0, min(100.0, score))

        # 디버그 정보
        debug_info = {
            'face_idx': face_idx,
            'tone_idx': tone_idx,
            'normalized_style': normalized,
            'style_found': normalized in self.style_to_idx,
            'raw_score': score_tensor.item(),
            'final_score': score,
            'combination': f"{face_shape}_{skin_tone}"
        }

        return score, debug_info

    def predict_batch(
        self,
        face_shapes: List[str],
        skin_tones: List[str],
        hairstyles: List[str]
    ) -> np.ndarray:
        """
        배치 예측

        Args:
            face_shapes: 얼굴형 리스트
            skin_tones: 피부톤 리스트
            hairstyles: 헤어스타일 리스트

        Returns:
            점수 배열
        """
        # 인덱스 변환
        face_indices = [self.face_to_idx.get(f, 3) for f in face_shapes]
        tone_indices = [self.tone_to_idx.get(t, 2) for t in skin_tones]

        # 스타일 임베딩
        from utils.style_preprocessor import normalize_style_name
        style_embeddings = []

        for style in hairstyles:
            normalized = normalize_style_name(style)
            if normalized in self.style_to_idx:
                idx = self.style_to_idx[normalized]
                style_embeddings.append(self.embeddings[idx])
            else:
                # 평균 임베딩 사용
                style_embeddings.append(np.mean(self.embeddings, axis=0))

        # 텐서 변환
        face_tensor = torch.tensor(face_indices, dtype=torch.long).to(self.device)
        tone_tensor = torch.tensor(tone_indices, dtype=torch.long).to(self.device)
        style_tensor = torch.tensor(style_embeddings, dtype=torch.float32).to(self.device)

        # 예측
        with torch.no_grad():
            scores = self.model(face_tensor, tone_tensor, style_tensor)
            scores = scores.cpu().numpy().flatten()

        # 범위 제한
        scores = np.clip(scores, 0, 100)

        return scores

    def get_embedding_weights(self) -> Dict:
        """
        학습된 임베딩 가중치 반환 (분석용)

        Returns:
            Face/Tone 임베딩 가중치
        """
        with torch.no_grad():
            face_weights = self.model.face_embedding.weight.cpu().numpy()
            tone_weights = self.model.tone_embedding.weight.cpu().numpy()

        return {
            'face_embeddings': face_weights,
            'tone_embeddings': tone_weights,
            'face_shapes': self.FACE_SHAPES,
            'skin_tones': self.SKIN_TONES
        }

    def analyze_combination_similarity(self) -> np.ndarray:
        """
        학습된 임베딩 기반 조합 간 유사도 분석

        Returns:
            유사도 매트릭스
        """
        embeddings = self.get_embedding_weights()
        face_emb = embeddings['face_embeddings']
        tone_emb = embeddings['tone_embeddings']

        # 모든 조합의 임베딩 생성
        combinations = []
        combo_names = []

        for i, face in enumerate(self.FACE_SHAPES):
            for j, tone in enumerate(self.SKIN_TONES):
                combo_emb = np.concatenate([face_emb[i], tone_emb[j]])
                combinations.append(combo_emb)
                combo_names.append(f"{face}_{tone}")

        combinations = np.array(combinations)

        # 코사인 유사도 계산
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(combinations)

        return similarity_matrix, combo_names


if __name__ == "__main__":
    # 테스트
    recommender = MLHairstyleRecommenderV3()

    # 단일 예측 테스트
    score, info = recommender.predict_score("계란형", "봄웜", "댄디 컷")
    print(f"Score: {score:.1f}")
    print(f"Debug: {info}")

    # 배치 예측 테스트
    faces = ["각진형", "둥근형", "긴형"]
    tones = ["겨울쿨", "가을웜", "여름쿨"]
    styles = ["댄디 컷", "투 블럭 컷", "리젠트 컷"]

    scores = recommender.predict_batch(faces, tones, styles)
    print(f"Batch scores: {scores}")