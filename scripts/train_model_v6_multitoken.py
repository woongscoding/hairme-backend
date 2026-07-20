#!/usr/bin/env python3
"""
v6 모델 학습 스크립트 - Multi-Token Attention 버전

**핵심 개선 (v5 대비):**
- 기존: 단일 토큰(480차원)에 self-attention → 사실상 identity (무의미)
- 개선: face/skin/style 3개 토큰 간 cross-attention → 의미 있는 상호작용 학습

**구조:**
1. Input Projection: face(6→64), skin(2→32)
2. Multi-Token Attention: 3개 토큰을 128차원으로 통일 후 self-attention
3. Feature Fusion: attention 출력(384) → MLP → score

**라벨 정규화 (v5 동일):**
- 원본 점수 범위: 10~95점
- normalized_label = (label - 10) / 85
- 역변환: final_score = model_output * 85 + 10

Author: HairMe ML Team
Date: 2025-12-02
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
from pathlib import Path
import logging
from typing import Dict, Tuple, List
from datetime import datetime
import json

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ========== 라벨 정규화 상수 ==========
LABEL_MIN = 10.0  # 원본 점수 최소값
LABEL_MAX = 95.0  # 원본 점수 최대값
LABEL_RANGE = LABEL_MAX - LABEL_MIN  # 85


def normalize_score(score: np.ndarray) -> np.ndarray:
    """원본 점수를 0~1로 정규화"""
    return (score - LABEL_MIN) / LABEL_RANGE


def denormalize_score(normalized: np.ndarray) -> np.ndarray:
    """0~1 점수를 원본 스케일로 역변환"""
    return normalized * LABEL_RANGE + LABEL_MIN


# ========== Multi-Token Attention Layer ==========
class MultiTokenAttentionLayer(nn.Module):
    """
    3-Token Cross-Attention Layer

    face_proj, skin_proj, style_emb를 3개의 개별 토큰으로 구성하여
    토큰 간의 상호작용을 학습합니다.
    """

    def __init__(
        self,
        face_dim: int = 64,
        skin_dim: int = 32,
        style_dim: int = 384,
        token_dim: int = 128,
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


# ========== 모델 정의 (V6) ==========
class RecommendationModelV6(nn.Module):
    """
    Multi-Token Attention 기반 추천 모델 v6

    핵심 특징:
    - 3개 토큰(face, skin, style) 간 cross-attention
    - 출력층에 Sigmoid 활성화 함수 사용 (0~1 출력 보장)
    - 추론 시 역변환 필요 (0~1 → 10~95)
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


# ========== 데이터셋 ==========
class HairstyleDatasetV6(Dataset):
    """정규화된 라벨 기반 헤어스타일 데이터셋"""

    def __init__(
        self,
        face_features: np.ndarray,
        skin_features: np.ndarray,
        style_embeddings: np.ndarray,
        scores: np.ndarray,
        normalize_labels: bool = True,
    ):
        self.face_features = torch.tensor(face_features, dtype=torch.float32)
        self.skin_features = torch.tensor(skin_features, dtype=torch.float32)
        self.style_embeddings = torch.tensor(style_embeddings, dtype=torch.float32)

        if normalize_labels:
            normalized_scores = normalize_score(scores)
            self.scores = torch.tensor(
                normalized_scores, dtype=torch.float32
            ).unsqueeze(1)
            logger.info(
                f"  라벨 정규화 적용: {scores.min():.1f}~{scores.max():.1f} → {normalized_scores.min():.3f}~{normalized_scores.max():.3f}"
            )
        else:
            self.scores = torch.tensor(scores, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return (
            self.face_features[idx],
            self.skin_features[idx],
            self.style_embeddings[idx],
            self.scores[idx],
        )


# ========== Face Group Batch Sampler ==========
class FaceGroupBatchSampler(Sampler):
    """
    같은 얼굴의 샘플들을 항상 같은 배치에 포함시키는 커스텀 배치 샘플러.

    Pairwise Ranking Loss 계산을 위해, 같은 얼굴의 6개 헤어스타일이
    동일 배치에 들어가야 쌍별 비교가 가능합니다.
    """

    def __init__(
        self,
        sample_indices: List[int],
        samples_per_face: int = 6,
        batch_size: int = 64,
        shuffle: bool = True,
    ):
        if samples_per_face < 2:
            raise ValueError("samples_per_face must be at least 2")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(sample_indices) % samples_per_face != 0:
            raise ValueError(
                "sample_indices length must be divisible by samples_per_face; "
                "incomplete face groups are not allowed"
            )

        self.samples_per_face = samples_per_face
        self.shuffle = shuffle

        # sample_indices를 얼굴 그룹으로 묶기
        self.face_groups = []
        for i in range(0, len(sample_indices), samples_per_face):
            group = sample_indices[i : i + samples_per_face]
            # 그룹 무결성 검증: 한 그룹의 인덱스는 전부 같은 얼굴이어야 함
            face_ids = {idx // samples_per_face for idx in group}
            if len(face_ids) != 1:
                raise ValueError(
                    f"FaceGroupBatchSampler 그룹 무결성 위반: 인덱스 {group}가 "
                    f"서로 다른 얼굴 {sorted(face_ids)}에 걸쳐 있습니다. "
                    f"sample_indices는 얼굴별로 연속된 {samples_per_face}개 "
                    f"단위로 정렬되어 있어야 합니다."
                )
            self.face_groups.append(group)

        # 배치당 얼굴 수 (batch_size를 samples_per_face의 배수로 조정)
        self.faces_per_batch = max(1, batch_size // samples_per_face)
        self.num_faces = len(self.face_groups)

    def __iter__(self):
        face_order = list(range(self.num_faces))
        if self.shuffle:
            np.random.shuffle(face_order)

        for i in range(0, self.num_faces, self.faces_per_batch):
            batch_faces = face_order[i : i + self.faces_per_batch]
            batch_indices = []
            for face_idx in batch_faces:
                batch_indices.extend(self.face_groups[face_idx])
            yield batch_indices

    def __len__(self):
        return (self.num_faces + self.faces_per_batch - 1) // self.faces_per_batch


# ========== Pairwise Ranking Loss ==========
def pairwise_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    samples_per_face: int = 6,
    margin: float = 0.05,
) -> torch.Tensor:
    """
    같은 얼굴의 헤어스타일 간 순위를 보존하는 Pairwise Ranking Loss.

    각 얼굴의 samples_per_face개 헤어스타일에서 C(n,2) 쌍을 생성하고,
    target_i > target_j인 쌍에 대해:
        loss = max(0, margin - (pred_i - pred_j))

    Args:
        predictions: (batch_size,) 모델 예측값
        targets: (batch_size,) or (batch_size, 1) 타겟값
        samples_per_face: 얼굴당 헤어스타일 수
        margin: 순위 마진 (정규화 스케일)

    Returns:
        scalar loss
    """
    targets = targets.view(-1)
    predictions = predictions.view(-1)

    batch_size = predictions.size(0)
    num_faces = batch_size // samples_per_face

    if num_faces == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    # 유효한 샘플만 사용 (samples_per_face의 배수)
    valid_size = num_faces * samples_per_face
    pred = predictions[:valid_size].view(num_faces, samples_per_face)
    tgt = targets[:valid_size].view(num_faces, samples_per_face)

    # C(samples_per_face, 2) 쌍의 인덱스 생성 (벡터화)
    idx_i, idx_j = torch.triu_indices(samples_per_face, samples_per_face, offset=1)

    # 모든 얼굴에 대해 한번에 쌍 추출: (num_faces, num_pairs)
    pred_i = pred[:, idx_i]
    pred_j = pred[:, idx_j]
    tgt_i = tgt[:, idx_i]
    tgt_j = tgt[:, idx_j]

    # target_i > target_j인 쌍: pred_i가 더 높아야 함
    # target_i < target_j인 쌍: pred_j가 더 높아야 함 (방향 반전)
    diff_target = tgt_i - tgt_j
    diff_pred = pred_i - pred_j

    # 타겟 차이의 부호에 따라 예측 차이의 방향 조정
    # sign(diff_target) * diff_pred가 양수여야 순위가 일치
    signed_diff = torch.sign(diff_target) * diff_pred

    # 동점(diff_target == 0)인 쌍은 제외
    non_tie_mask = diff_target.abs() > 1e-6

    if non_tie_mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    # Margin ranking loss: max(0, margin - signed_diff)
    losses = torch.clamp(margin - signed_diff, min=0.0)
    loss = losses[non_tie_mask].mean()

    return loss


def compute_pairwise_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    samples_per_face: int = 6,
) -> float:
    """
    쌍별 순위 정확도 계산.

    target_i > target_j일 때 pred_i > pred_j인 비율을 계산합니다.

    Returns:
        정확도 (0.0 ~ 1.0)
    """
    targets = targets.view(-1)
    predictions = predictions.view(-1)

    batch_size = predictions.size(0)
    num_faces = batch_size // samples_per_face

    if num_faces == 0:
        return 0.0

    valid_size = num_faces * samples_per_face
    pred = predictions[:valid_size].view(num_faces, samples_per_face)
    tgt = targets[:valid_size].view(num_faces, samples_per_face)

    idx_i, idx_j = torch.triu_indices(samples_per_face, samples_per_face, offset=1)

    pred_i = pred[:, idx_i]
    pred_j = pred[:, idx_j]
    tgt_i = tgt[:, idx_i]
    tgt_j = tgt[:, idx_j]

    diff_target = tgt_i - tgt_j
    diff_pred = pred_i - pred_j

    # 동점 제외
    non_tie_mask = diff_target.abs() > 1e-6

    if non_tie_mask.sum() == 0:
        return 1.0  # 모든 쌍이 동점이면 100%

    # 순위 일치: sign(diff_target) == sign(diff_pred)
    correct = (torch.sign(diff_target) == torch.sign(diff_pred)) & non_tie_mask
    accuracy = correct.sum().float() / non_tie_mask.sum().float()

    return accuracy.item()


def load_training_data(data_path: str) -> Dict:
    """NPZ 데이터 로드 및 전처리"""
    logger.info(f"📂 데이터 로딩: {data_path}")

    data = np.load(data_path, allow_pickle=True)

    face_features = data["face_features"]
    skin_features = data["skin_features"]
    hairstyles = data["hairstyles"]
    scores = data["scores"]

    logger.info(f"✅ 데이터 로드 완료:")
    logger.info(f"  - 샘플 수: {len(scores):,}")
    logger.info(f"  - Face features: {face_features.shape}")
    logger.info(f"  - Skin features: {skin_features.shape}")
    logger.info(f"  - Hairstyles: {len(hairstyles)}")

    # 헤어스타일 임베딩 생성
    logger.info("🔄 헤어스타일 임베딩 생성 중...")
    sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    style_embeddings = sentence_model.encode(
        hairstyles.tolist(), show_progress_bar=True, convert_to_numpy=True
    )

    logger.info(f"✅ 임베딩 생성 완료: {style_embeddings.shape}")

    # 통계
    logger.info(f"\n📊 원본 데이터 통계:")
    logger.info(f"  - 점수 범위: {scores.min():.1f} ~ {scores.max():.1f}")
    logger.info(f"  - 점수 평균: {scores.mean():.1f} ± {scores.std():.1f}")
    logger.info(
        f"  - 고점수 (≥75): {(scores >= 75).sum():,}개 ({(scores >= 75).sum()/len(scores)*100:.1f}%)"
    )
    logger.info(
        f"  - 저점수 (≤40): {(scores <= 40).sum():,}개 ({(scores <= 40).sum()/len(scores)*100:.1f}%)"
    )

    return {
        "face_features": face_features,
        "skin_features": skin_features,
        "style_embeddings": style_embeddings,
        "scores": scores,
        "hairstyles": hairstyles,
    }


def create_dataloaders_no_leakage(
    data: Dict,
    batch_size: int = 64,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    samples_per_face: int = 6,
    use_face_group_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """데이터 리키지 방지: 얼굴 단위로 train/val/test 분할

    Args:
        use_face_group_sampler: True이면 FaceGroupBatchSampler를 사용하여
            같은 얼굴의 샘플을 같은 배치에 배치 (Ranking Loss용)
    """

    dataset = HairstyleDatasetV6(
        face_features=data["face_features"],
        skin_features=data["skin_features"],
        style_embeddings=data["style_embeddings"],
        scores=data["scores"],
        normalize_labels=True,
    )

    total_samples = len(dataset)
    if samples_per_face < 2:
        raise ValueError("samples_per_face must be at least 2")
    if total_samples % samples_per_face != 0:
        raise ValueError(
            "total samples must be divisible by samples_per_face; "
            "incomplete face groups are not allowed"
        )

    num_faces = total_samples // samples_per_face

    logger.info(f"\n🔍 데이터 리키지 방지 모드:")
    logger.info(f"  - 총 샘플: {total_samples:,}개")
    logger.info(f"  - 총 얼굴: {num_faces:,}개")
    logger.info(f"  - 얼굴당 샘플: {samples_per_face}개")
    if use_face_group_sampler:
        logger.info(f"  - Face Group Sampler 활성화 (Ranking Loss용)")

    np.random.seed(42)
    face_indices = np.random.permutation(num_faces)

    num_test_faces = int(num_faces * test_ratio)
    num_val_faces = int(num_faces * val_ratio)
    num_train_faces = num_faces - num_test_faces - num_val_faces
    if min(num_train_faces, num_val_faces, num_test_faces) < 1:
        raise ValueError(
            "train/validation/test splits must each contain at least one face; "
            "increase the dataset size or adjust val_ratio/test_ratio"
        )

    train_face_indices = face_indices[:num_train_faces]
    val_face_indices = face_indices[num_train_faces : num_train_faces + num_val_faces]
    test_face_indices = face_indices[num_train_faces + num_val_faces :]

    logger.info(f"\n📊 얼굴 단위 분할:")
    logger.info(
        f"  - Train: {num_train_faces:,}개 얼굴 ({num_train_faces/num_faces*100:.1f}%)"
    )
    logger.info(
        f"  - Val:   {num_val_faces:,}개 얼굴 ({num_val_faces/num_faces*100:.1f}%)"
    )
    logger.info(
        f"  - Test:  {num_test_faces:,}개 얼굴 ({num_test_faces/num_faces*100:.1f}%)"
    )

    def face_indices_to_sample_indices(face_idxs: np.ndarray) -> List[int]:
        sample_idxs = []
        for face_idx in face_idxs:
            start = face_idx * samples_per_face
            end = start + samples_per_face
            sample_idxs.extend(range(start, end))
        return sample_idxs

    train_sample_indices = face_indices_to_sample_indices(train_face_indices)
    val_sample_indices = face_indices_to_sample_indices(val_face_indices)
    test_sample_indices = face_indices_to_sample_indices(test_face_indices)

    train_dataset = Subset(dataset, train_sample_indices)
    val_dataset = Subset(dataset, val_sample_indices)
    test_dataset = Subset(dataset, test_sample_indices)

    # val/test 로더는 rank_weight와 무관하게 항상 FaceGroupBatchSampler(shuffle=False)를
    # 사용한다. validate()가 항상 6개 단위 reshape로 pairwise 지표를 계산하므로,
    # 배치 경계가 반드시 samples_per_face의 배수로 정렬되어야 서로 다른 얼굴을 섞어
    # 비교하는 오프셋 버그가 발생하지 않는다. (val/test 인덱스는 이미 얼굴별 연속)
    val_sampler = FaceGroupBatchSampler(
        val_sample_indices,
        samples_per_face=samples_per_face,
        batch_size=batch_size,
        shuffle=False,
    )
    test_sampler = FaceGroupBatchSampler(
        test_sample_indices,
        samples_per_face=samples_per_face,
        batch_size=batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        dataset,
        batch_sampler=val_sampler,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset,
        batch_sampler=test_sampler,
        num_workers=0,
        pin_memory=True,
    )

    # train 로더는 기존 로직 유지: rank_weight>0(=use_face_group_sampler)일 때만
    # FaceGroupBatchSampler를 사용하고, 그 외에는 일반 셔플 DataLoader를 사용한다.
    if use_face_group_sampler:
        train_sampler = FaceGroupBatchSampler(
            train_sample_indices,
            samples_per_face=samples_per_face,
            batch_size=batch_size,
            shuffle=True,
        )
        train_loader = DataLoader(
            dataset,
            batch_sampler=train_sampler,
            num_workers=0,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    rank_weight: float = 0.0,
    rank_margin: float = 0.05,
    samples_per_face: int = 6,
) -> Dict[str, float]:
    """1 에폭 학습

    Returns:
        dict with keys: total_loss, mse_loss, rank_loss
    """
    model.train()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_rank_loss = 0.0

    for face_feat, skin_feat, style_emb, scores in train_loader:
        face_feat = face_feat.to(device)
        skin_feat = skin_feat.to(device)
        style_emb = style_emb.to(device)
        scores = scores.to(device)

        optimizer.zero_grad()
        pred_scores = model(face_feat, skin_feat, style_emb)

        mse_loss = criterion(pred_scores.unsqueeze(1), scores)

        if rank_weight > 0:
            rank_loss = pairwise_ranking_loss(
                pred_scores,
                scores,
                samples_per_face=samples_per_face,
                margin=rank_margin,
            )
            loss = mse_loss + rank_weight * rank_loss
            total_rank_loss += rank_loss.item()
        else:
            loss = mse_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse_loss += mse_loss.item()

    num_batches = len(train_loader)
    return {
        "total_loss": total_loss / num_batches,
        "mse_loss": total_mse_loss / num_batches,
        "rank_loss": total_rank_loss / num_batches if rank_weight > 0 else 0.0,
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    rank_weight: float = 0.0,
    rank_margin: float = 0.05,
    samples_per_face: int = 6,
) -> Dict[str, float]:
    """검증

    Returns:
        dict with keys: total_loss, mse_loss, rank_loss, mae_normalized, mae_original, pairwise_accuracy
    """
    model.eval()
    total_mse_loss = 0.0
    total_rank_loss = 0.0
    total_normalized_mae = 0.0
    total_original_mae = 0.0
    total_pairwise_acc = 0.0
    num_rank_batches = 0

    with torch.no_grad():
        for face_feat, skin_feat, style_emb, scores in val_loader:
            face_feat = face_feat.to(device)
            skin_feat = skin_feat.to(device)
            style_emb = style_emb.to(device)
            scores = scores.to(device)

            pred_scores = model(face_feat, skin_feat, style_emb)

            mse_loss = criterion(pred_scores.unsqueeze(1), scores)
            normalized_mae = torch.abs(pred_scores.unsqueeze(1) - scores).mean()

            pred_original = pred_scores.unsqueeze(1) * LABEL_RANGE + LABEL_MIN
            scores_original = scores * LABEL_RANGE + LABEL_MIN
            original_mae = torch.abs(pred_original - scores_original).mean()

            total_mse_loss += mse_loss.item()
            total_normalized_mae += normalized_mae.item()
            total_original_mae += original_mae.item()

            # Ranking metrics (항상 계산, 모니터링용)
            batch_size = pred_scores.size(0)
            if batch_size >= samples_per_face:
                rank_loss = pairwise_ranking_loss(
                    pred_scores,
                    scores,
                    samples_per_face=samples_per_face,
                    margin=rank_margin,
                )
                pair_acc = compute_pairwise_accuracy(
                    pred_scores, scores, samples_per_face=samples_per_face
                )
                total_rank_loss += rank_loss.item()
                total_pairwise_acc += pair_acc
                num_rank_batches += 1

    num_batches = len(val_loader)

    # ranking 배치가 하나도 없으면 지표를 0.0으로 위장하지 않고 NaN으로 반환한다.
    # (best model 선택은 mse/total_loss 기반이므로 NaN이 선택 기준에 새어들면 안 됨)
    if num_rank_batches > 0:
        rank_loss = total_rank_loss / num_rank_batches
        pairwise_accuracy = total_pairwise_acc / num_rank_batches
    else:
        rank_loss = float("nan")
        pairwise_accuracy = float("nan")

    # total_loss는 항상 유한해야 한다: ranking 배치가 없을 때는 ranking 항을 더하지 않는다.
    total_loss = total_mse_loss / num_batches
    if rank_weight > 0 and num_rank_batches > 0:
        total_loss += rank_weight * rank_loss

    return {
        "total_loss": total_loss,
        "mse_loss": total_mse_loss / num_batches,
        "rank_loss": rank_loss,
        "mae_normalized": total_normalized_mae / num_batches,
        "mae_original": total_original_mae / num_batches,
        "pairwise_accuracy": pairwise_accuracy,
    }


def count_parameters(model: nn.Module) -> int:
    """모델 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _fmt_metric(value: float, fmt: str = ".4f") -> str:
    """지표 포맷팅. NaN(ranking 배치 없음)이면 'N/A'로 표시."""
    if value != value:  # NaN 체크
        return "N/A"
    return format(value, fmt)


def train_model(
    data_path: str,
    output_dir: str = "models",
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 15,
    token_dim: int = 128,
    num_heads: int = 4,
    rank_weight: float = 0.0,
    rank_margin: float = 0.05,
):
    """모델 학습 메인 함수 (V6 - Multi-Token Attention)"""
    logger.info("=" * 60)
    logger.info("🚀 V6 모델 학습 시작 (Multi-Token Attention)")
    logger.info("=" * 60)
    logger.info(f"  - 핵심 개선: 3개 토큰 간 Cross-Attention")
    logger.info(f"  - Token dimension: {token_dim}")
    logger.info(f"  - Attention heads: {num_heads}")
    logger.info(f"  - 라벨 정규화: {LABEL_MIN}~{LABEL_MAX} → 0~1")
    if rank_weight > 0:
        logger.info(f"  - Ranking Loss: weight={rank_weight}, margin={rank_margin}")
    else:
        logger.info(f"  - Ranking Loss: 비활성화 (rank_weight=0)")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  디바이스: {device}")

    # 데이터 로드
    data = load_training_data(data_path)

    # 데이터로더 생성 (ranking 활성화 시 FaceGroupBatchSampler 사용)
    use_face_group = rank_weight > 0
    train_loader, val_loader, test_loader = create_dataloaders_no_leakage(
        data, batch_size=batch_size, use_face_group_sampler=use_face_group
    )

    # V6 모델 생성
    model = RecommendationModelV6(
        face_feat_dim=6,
        skin_feat_dim=2,
        style_embed_dim=384,
        token_dim=token_dim,
        num_heads=num_heads,
        dropout_rate=0.3,
    ).to(device)

    num_params = count_parameters(model)

    logger.info(f"\n🏗️  모델 구조 (V6 - Multi-Token Attention):")
    logger.info(f"  - Face features: 6 → 64")
    logger.info(f"  - Skin features: 2 → 32")
    logger.info(f"  - Style embedding: 384")
    logger.info(f"  - Token dimension: {token_dim}")
    logger.info(f"  - Attention: 3-token cross-attention ({num_heads} heads)")
    logger.info(f"  - Attention output: {token_dim * 3}")
    logger.info(f"  - 출력층: Sigmoid (0~1)")
    logger.info(f"  - 총 파라미터: {num_params:,}")

    # 학습 설정
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = {
        "train_loss": [],
        "train_mse_loss": [],
        "train_rank_loss": [],
        "val_loss": [],
        "val_mse_loss": [],
        "val_rank_loss": [],
        "val_mae_normalized": [],
        "val_mae_original": [],
        "val_pairwise_accuracy": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    logger.info(f"\n🏋️  학습 시작:")
    logger.info(f"  - Epochs: {epochs}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Learning rate: {learning_rate}")
    logger.info(f"  - Early stopping patience: {patience}")
    if rank_weight > 0:
        logger.info(
            f"  - Loss: MSE + {rank_weight} * RankingLoss (margin={rank_margin})"
        )

    for epoch in range(epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            rank_weight=rank_weight,
            rank_margin=rank_margin,
        )
        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            rank_weight=rank_weight,
            rank_margin=rank_margin,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["total_loss"])
        history["train_mse_loss"].append(train_metrics["mse_loss"])
        history["train_rank_loss"].append(train_metrics["rank_loss"])
        history["val_loss"].append(val_metrics["total_loss"])
        history["val_mse_loss"].append(val_metrics["mse_loss"])
        history["val_rank_loss"].append(val_metrics["rank_loss"])
        history["val_mae_normalized"].append(val_metrics["mae_normalized"])
        history["val_mae_original"].append(val_metrics["mae_original"])
        history["val_pairwise_accuracy"].append(val_metrics["pairwise_accuracy"])
        history["lr"].append(current_lr)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            log_msg = (
                f"Epoch [{epoch+1:3d}/{epochs}] "
                f"Train Loss: {train_metrics['total_loss']:.4f} | "
                f"Val Loss: {val_metrics['total_loss']:.4f} | "
                f"MAE(orig): {val_metrics['mae_original']:.2f}"
            )
            if rank_weight > 0:
                log_msg += (
                    f" | RankLoss: {_fmt_metric(val_metrics['rank_loss'])}"
                    f" | PairAcc: {_fmt_metric(val_metrics['pairwise_accuracy'], '.3f')}"
                )
            log_msg += f" | LR: {current_lr:.6f}"
            logger.info(log_msg)

        scheduler.step(val_metrics["total_loss"])

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            patience_counter = 0

            model_path = output_dir / "hairstyle_recommender_v6_multitoken.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                    "config": {
                        "version": "v6",
                        "face_feat_dim": 6,
                        "skin_feat_dim": 2,
                        "style_embed_dim": 384,
                        "token_dim": token_dim,
                        "num_heads": num_heads,
                        "normalized": True,
                        "label_min": LABEL_MIN,
                        "label_max": LABEL_MAX,
                        "label_range": LABEL_RANGE,
                        "attention_type": "multi_token",
                        "rank_weight": rank_weight,
                        "rank_margin": rank_margin,
                    },
                },
                model_path,
            )

            logger.info(f"  ✅ Best 모델 저장: {model_path}")

        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"\n⏸️  Early stopping at epoch {epoch + 1}")
            break

    # 최종 테스트: early stopping 시점의 마지막 모델이 아니라 저장된 best 모델로 평가
    best_model_path = output_dir / "hairstyle_recommender_v6_multitoken.pt"
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"\n♻️  Best 모델 로드 (epoch {checkpoint['epoch']}, "
            f"val_loss {checkpoint['best_val_loss']:.4f})"
        )

    logger.info(f"\n🧪 최종 테스트 평가:")
    test_metrics = validate(
        model,
        test_loader,
        criterion,
        device,
        rank_weight=rank_weight,
        rank_margin=rank_margin,
    )
    logger.info(f"  - Test Loss: {test_metrics['total_loss']:.4f}")
    logger.info(f"  - Test MSE Loss: {test_metrics['mse_loss']:.4f}")
    logger.info(f"  - Test MAE (정규화): {test_metrics['mae_normalized']:.4f}")
    logger.info(f"  - Test MAE (원본 스케일): {test_metrics['mae_original']:.2f}점")
    if rank_weight > 0:
        logger.info(f"  - Test Ranking Loss: {_fmt_metric(test_metrics['rank_loss'])}")
    logger.info(
        f"  - Test Pairwise Accuracy: {_fmt_metric(test_metrics['pairwise_accuracy'], '.3f')}"
    )

    # 학습 기록 저장 (NaN은 JSON 표준에 없어 strict 파서를 깨뜨리므로 null로 변환)
    history_path = output_dir / "training_history_v6_multitoken.json"
    history_safe = {
        key: [None if value != value else value for value in values]
        for key, values in history.items()
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_safe, f, indent=2)

    logger.info(f"\n📊 학습 기록 저장: {history_path}")

    # 모델 출력 범위 검증
    logger.info(f"\n🔍 모델 출력 범위 검증:")
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for face_feat, skin_feat, style_emb, scores in test_loader:
            face_feat = face_feat.to(device)
            skin_feat = skin_feat.to(device)
            style_emb = style_emb.to(device)

            pred = model(face_feat, skin_feat, style_emb)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(scores.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels).flatten()

        logger.info(
            f"  - 정규화된 출력 범위: {all_preds.min():.4f} ~ {all_preds.max():.4f}"
        )
        logger.info(
            f"  - 정규화된 라벨 범위: {all_labels.min():.4f} ~ {all_labels.max():.4f}"
        )

        preds_orig = all_preds * LABEL_RANGE + LABEL_MIN
        labels_orig = all_labels * LABEL_RANGE + LABEL_MIN

        logger.info(
            f"  - 원본 스케일 출력 범위: {preds_orig.min():.1f} ~ {preds_orig.max():.1f}"
        )
        logger.info(
            f"  - 원본 스케일 라벨 범위: {labels_orig.min():.1f} ~ {labels_orig.max():.1f}"
        )

        # 분류 정확도
        high_threshold = normalize_score(np.array([75.0]))[0]
        low_threshold = normalize_score(np.array([40.0]))[0]

        high_labels = all_labels >= high_threshold
        low_labels = all_labels <= low_threshold

        if high_labels.sum() > 0:
            high_correct = (all_preds[high_labels] >= high_threshold).sum()
            logger.info(f"\n📊 분류 정확도:")
            logger.info(
                f"  - 고점수(≥75점) 예측 정확도: {high_correct}/{high_labels.sum()} ({high_correct/high_labels.sum()*100:.1f}%)"
            )

        if low_labels.sum() > 0:
            low_correct = (all_preds[low_labels] <= low_threshold).sum()
            logger.info(
                f"  - 저점수(≤40점) 예측 정확도: {low_correct}/{low_labels.sum()} ({low_correct/low_labels.sum()*100:.1f}%)"
            )

    logger.info("\n" + "=" * 60)
    logger.info("✅ V6 학습 완료!")
    logger.info("=" * 60)
    logger.info(f"  - Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"  - Test MAE: {test_metrics['mae_original']:.2f}점 (원본 스케일)")
    if rank_weight > 0:
        logger.info(
            f"  - Test Pairwise Accuracy: {_fmt_metric(test_metrics['pairwise_accuracy'], '.3f')}"
        )
    logger.info(
        f"  - 모델 경로: {output_dir / 'hairstyle_recommender_v6_multitoken.pt'}"
    )


def main():
    parser = argparse.ArgumentParser(description="V6 모델 학습 (Multi-Token Attention)")
    parser.add_argument(
        "--data",
        type=str,
        default="data_source/ai_face_1000.npz",
        help="학습 데이터 NPZ 경로",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models", help="모델 저장 디렉토리"
    )
    parser.add_argument("--epochs", type=int, default=100, help="학습 에폭 수")
    parser.add_argument("--batch-size", type=int, default=64, help="배치 크기")
    parser.add_argument("--lr", type=float, default=0.001, help="학습률")
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )
    parser.add_argument(
        "--token-dim", type=int, default=128, help="Attention 토큰 차원"
    )
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads 수")
    parser.add_argument(
        "--rank-weight",
        type=float,
        default=0.0,
        help="Pairwise Ranking Loss 가중치 (0=비활성화, 1.0=MSE와 동일 가중치)",
    )
    parser.add_argument(
        "--rank-margin",
        type=float,
        default=0.05,
        help="Ranking Loss 마진 (정규화 스케일, 기본 0.05 ≈ 원본 4.25점)",
    )

    args = parser.parse_args()

    train_model(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        token_dim=args.token_dim,
        num_heads=args.num_heads,
        rank_weight=args.rank_weight,
        rank_margin=args.rank_margin,
    )


if __name__ == "__main__":
    main()
