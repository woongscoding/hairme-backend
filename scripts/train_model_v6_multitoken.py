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
from torch.utils.data import Dataset, DataLoader, Subset
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
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ========== 라벨 정규화 상수 ==========
LABEL_MIN = 10.0   # 원본 점수 최소값
LABEL_MAX = 95.0   # 원본 점수 최대값
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
        dropout: float = 0.1
    ):
        super().__init__()
        self.token_dim = token_dim

        # 각 입력을 동일한 token_dim으로 projection
        self.face_to_token = nn.Linear(face_dim, token_dim)
        self.skin_to_token = nn.Linear(skin_dim, token_dim)
        self.style_to_token = nn.Linear(style_dim, token_dim)

        # Multi-head self-attention (3 tokens)
        self.attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(token_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim)
        )
        self.norm2 = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        face_proj: torch.Tensor,  # (batch, 64)
        skin_proj: torch.Tensor,  # (batch, 32)
        style_emb: torch.Tensor   # (batch, 384)
    ) -> torch.Tensor:
        """
        Returns:
            (batch, token_dim * 3) - 3개 토큰의 concat 결과
        """
        batch_size = face_proj.size(0)

        # 각 특징을 token_dim으로 projection
        face_token = self.face_to_token(face_proj)   # (batch, token_dim)
        skin_token = self.skin_to_token(skin_proj)   # (batch, token_dim)
        style_token = self.style_to_token(style_emb) # (batch, token_dim)

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
        dropout_rate: float = 0.3
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
            nn.Dropout(dropout_rate * 0.5)
        )

        self.skin_projection = nn.Sequential(
            nn.Linear(skin_feat_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )

        # Multi-Token Attention Layer (3개 토큰 간 상호작용)
        self.multi_token_attention = MultiTokenAttentionLayer(
            face_dim=64,
            skin_dim=32,
            style_dim=style_embed_dim,
            token_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout_rate * 0.3
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
        style_emb: torch.Tensor
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
        normalize_labels: bool = True
    ):
        self.face_features = torch.tensor(face_features, dtype=torch.float32)
        self.skin_features = torch.tensor(skin_features, dtype=torch.float32)
        self.style_embeddings = torch.tensor(style_embeddings, dtype=torch.float32)

        if normalize_labels:
            normalized_scores = normalize_score(scores)
            self.scores = torch.tensor(normalized_scores, dtype=torch.float32).unsqueeze(1)
            logger.info(f"  라벨 정규화 적용: {scores.min():.1f}~{scores.max():.1f} → {normalized_scores.min():.3f}~{normalized_scores.max():.3f}")
        else:
            self.scores = torch.tensor(scores, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return (
            self.face_features[idx],
            self.skin_features[idx],
            self.style_embeddings[idx],
            self.scores[idx]
        )


def load_training_data(data_path: str) -> Dict:
    """NPZ 데이터 로드 및 전처리"""
    logger.info(f"📂 데이터 로딩: {data_path}")

    data = np.load(data_path, allow_pickle=True)

    face_features = data['face_features']
    skin_features = data['skin_features']
    hairstyles = data['hairstyles']
    scores = data['scores']

    logger.info(f"✅ 데이터 로드 완료:")
    logger.info(f"  - 샘플 수: {len(scores):,}")
    logger.info(f"  - Face features: {face_features.shape}")
    logger.info(f"  - Skin features: {skin_features.shape}")
    logger.info(f"  - Hairstyles: {len(hairstyles)}")

    # 헤어스타일 임베딩 생성
    logger.info("🔄 헤어스타일 임베딩 생성 중...")
    sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    style_embeddings = sentence_model.encode(
        hairstyles.tolist(),
        show_progress_bar=True,
        convert_to_numpy=True
    )

    logger.info(f"✅ 임베딩 생성 완료: {style_embeddings.shape}")

    # 통계
    logger.info(f"\n📊 원본 데이터 통계:")
    logger.info(f"  - 점수 범위: {scores.min():.1f} ~ {scores.max():.1f}")
    logger.info(f"  - 점수 평균: {scores.mean():.1f} ± {scores.std():.1f}")
    logger.info(f"  - 고점수 (≥75): {(scores >= 75).sum():,}개 ({(scores >= 75).sum()/len(scores)*100:.1f}%)")
    logger.info(f"  - 저점수 (≤40): {(scores <= 40).sum():,}개 ({(scores <= 40).sum()/len(scores)*100:.1f}%)")

    return {
        'face_features': face_features,
        'skin_features': skin_features,
        'style_embeddings': style_embeddings,
        'scores': scores,
        'hairstyles': hairstyles
    }


def create_dataloaders_no_leakage(
    data: Dict,
    batch_size: int = 64,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    samples_per_face: int = 6
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """데이터 리키지 방지: 얼굴 단위로 train/val/test 분할"""

    dataset = HairstyleDatasetV6(
        face_features=data['face_features'],
        skin_features=data['skin_features'],
        style_embeddings=data['style_embeddings'],
        scores=data['scores'],
        normalize_labels=True
    )

    total_samples = len(dataset)
    num_faces = total_samples // samples_per_face

    logger.info(f"\n🔍 데이터 리키지 방지 모드:")
    logger.info(f"  - 총 샘플: {total_samples:,}개")
    logger.info(f"  - 총 얼굴: {num_faces:,}개")
    logger.info(f"  - 얼굴당 샘플: {samples_per_face}개")

    np.random.seed(42)
    face_indices = np.random.permutation(num_faces)

    num_test_faces = int(num_faces * test_ratio)
    num_val_faces = int(num_faces * val_ratio)
    num_train_faces = num_faces - num_test_faces - num_val_faces

    train_face_indices = face_indices[:num_train_faces]
    val_face_indices = face_indices[num_train_faces:num_train_faces + num_val_faces]
    test_face_indices = face_indices[num_train_faces + num_val_faces:]

    logger.info(f"\n📊 얼굴 단위 분할:")
    logger.info(f"  - Train: {num_train_faces:,}개 얼굴 ({num_train_faces/num_faces*100:.1f}%)")
    logger.info(f"  - Val:   {num_val_faces:,}개 얼굴 ({num_val_faces/num_faces*100:.1f}%)")
    logger.info(f"  - Test:  {num_test_faces:,}개 얼굴 ({num_test_faces/num_faces*100:.1f}%)")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """1 에폭 학습"""
    model.train()
    total_loss = 0.0

    for face_feat, skin_feat, style_emb, scores in train_loader:
        face_feat = face_feat.to(device)
        skin_feat = skin_feat.to(device)
        style_emb = style_emb.to(device)
        scores = scores.to(device)

        optimizer.zero_grad()
        pred_scores = model(face_feat, skin_feat, style_emb)

        loss = criterion(pred_scores.unsqueeze(1), scores)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """검증"""
    model.eval()
    total_loss = 0.0
    total_normalized_mae = 0.0
    total_original_mae = 0.0

    with torch.no_grad():
        for face_feat, skin_feat, style_emb, scores in val_loader:
            face_feat = face_feat.to(device)
            skin_feat = skin_feat.to(device)
            style_emb = style_emb.to(device)
            scores = scores.to(device)

            pred_scores = model(face_feat, skin_feat, style_emb)

            loss = criterion(pred_scores.unsqueeze(1), scores)
            normalized_mae = torch.abs(pred_scores.unsqueeze(1) - scores).mean()

            pred_original = pred_scores.unsqueeze(1) * LABEL_RANGE + LABEL_MIN
            scores_original = scores * LABEL_RANGE + LABEL_MIN
            original_mae = torch.abs(pred_original - scores_original).mean()

            total_loss += loss.item()
            total_normalized_mae += normalized_mae.item()
            total_original_mae += original_mae.item()

    num_batches = len(val_loader)
    return (
        total_loss / num_batches,
        total_normalized_mae / num_batches,
        total_original_mae / num_batches
    )


def count_parameters(model: nn.Module) -> int:
    """모델 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    data_path: str,
    output_dir: str = "models",
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 15,
    token_dim: int = 128,
    num_heads: int = 4
):
    """모델 학습 메인 함수 (V6 - Multi-Token Attention)"""
    logger.info("=" * 60)
    logger.info("🚀 V6 모델 학습 시작 (Multi-Token Attention)")
    logger.info("=" * 60)
    logger.info(f"  - 핵심 개선: 3개 토큰 간 Cross-Attention")
    logger.info(f"  - Token dimension: {token_dim}")
    logger.info(f"  - Attention heads: {num_heads}")
    logger.info(f"  - 라벨 정규화: {LABEL_MIN}~{LABEL_MAX} → 0~1")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  디바이스: {device}")

    # 데이터 로드
    data = load_training_data(data_path)

    # 데이터로더 생성
    train_loader, val_loader, test_loader = create_dataloaders_no_leakage(
        data,
        batch_size=batch_size
    )

    # V6 모델 생성
    model = RecommendationModelV6(
        face_feat_dim=6,
        skin_feat_dim=2,
        style_embed_dim=384,
        token_dim=token_dim,
        num_heads=num_heads,
        dropout_rate=0.3
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
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae_normalized': [],
        'val_mae_original': [],
        'lr': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    logger.info(f"\n🏋️  학습 시작:")
    logger.info(f"  - Epochs: {epochs}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Learning rate: {learning_rate}")
    logger.info(f"  - Early stopping patience: {patience}")

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae_norm, val_mae_orig = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae_normalized'].append(val_mae_norm)
        history['val_mae_original'].append(val_mae_orig)
        history['lr'].append(current_lr)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch+1:3d}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"MAE(orig): {val_mae_orig:.2f} | "
                f"LR: {current_lr:.6f}"
            )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            model_path = output_dir / "hairstyle_recommender_v6_multitoken.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history,
                'config': {
                    'version': 'v6',
                    'face_feat_dim': 6,
                    'skin_feat_dim': 2,
                    'style_embed_dim': 384,
                    'token_dim': token_dim,
                    'num_heads': num_heads,
                    'normalized': True,
                    'label_min': LABEL_MIN,
                    'label_max': LABEL_MAX,
                    'label_range': LABEL_RANGE,
                    'attention_type': 'multi_token'
                }
            }, model_path)

            logger.info(f"  ✅ Best 모델 저장: {model_path}")

        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"\n⏸️  Early stopping at epoch {epoch + 1}")
            break

    # 최종 테스트
    logger.info(f"\n🧪 최종 테스트 평가:")
    test_loss, test_mae_norm, test_mae_orig = validate(model, test_loader, criterion, device)
    logger.info(f"  - Test Loss: {test_loss:.4f}")
    logger.info(f"  - Test MAE (정규화): {test_mae_norm:.4f}")
    logger.info(f"  - Test MAE (원본 스케일): {test_mae_orig:.2f}점")

    # 학습 기록 저장
    history_path = output_dir / "training_history_v6_multitoken.json"
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

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

        logger.info(f"  - 정규화된 출력 범위: {all_preds.min():.4f} ~ {all_preds.max():.4f}")
        logger.info(f"  - 정규화된 라벨 범위: {all_labels.min():.4f} ~ {all_labels.max():.4f}")

        preds_orig = all_preds * LABEL_RANGE + LABEL_MIN
        labels_orig = all_labels * LABEL_RANGE + LABEL_MIN

        logger.info(f"  - 원본 스케일 출력 범위: {preds_orig.min():.1f} ~ {preds_orig.max():.1f}")
        logger.info(f"  - 원본 스케일 라벨 범위: {labels_orig.min():.1f} ~ {labels_orig.max():.1f}")

        # 분류 정확도
        high_threshold = normalize_score(np.array([75.0]))[0]
        low_threshold = normalize_score(np.array([40.0]))[0]

        high_labels = all_labels >= high_threshold
        low_labels = all_labels <= low_threshold

        if high_labels.sum() > 0:
            high_correct = (all_preds[high_labels] >= high_threshold).sum()
            logger.info(f"\n📊 분류 정확도:")
            logger.info(f"  - 고점수(≥75점) 예측 정확도: {high_correct}/{high_labels.sum()} ({high_correct/high_labels.sum()*100:.1f}%)")

        if low_labels.sum() > 0:
            low_correct = (all_preds[low_labels] <= low_threshold).sum()
            logger.info(f"  - 저점수(≤40점) 예측 정확도: {low_correct}/{low_labels.sum()} ({low_correct/low_labels.sum()*100:.1f}%)")

    logger.info("\n" + "=" * 60)
    logger.info("✅ V6 학습 완료!")
    logger.info("=" * 60)
    logger.info(f"  - Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"  - Test MAE: {test_mae_orig:.2f}점 (원본 스케일)")
    logger.info(f"  - 모델 경로: {output_dir / 'hairstyle_recommender_v6_multitoken.pt'}")


def main():
    parser = argparse.ArgumentParser(description="V6 모델 학습 (Multi-Token Attention)")
    parser.add_argument(
        "--data",
        type=str,
        default="data_source/ai_face_1000.npz",
        help="학습 데이터 NPZ 경로"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="모델 저장 디렉토리"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="학습 에폭 수"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="배치 크기"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="학습률"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--token-dim",
        type=int,
        default=128,
        help="Attention 토큰 차원"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Attention heads 수"
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
        num_heads=args.num_heads
    )


if __name__ == "__main__":
    main()
