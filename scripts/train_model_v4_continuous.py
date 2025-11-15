#!/usr/bin/env python3
"""
v4 모델 학습 스크립트 - 연속형 변수 기반

입력 데이터 형식 (NPZ):
- face_features: [N, 6] - MediaPipe 얼굴 측정값
- skin_features: [N, 2] - MediaPipe 피부 측정값
- hairstyles: [N] - 헤어스타일명
- scores: [N] - 추천 점수 (0-100)

Author: HairMe ML Team
Date: 2025-11-15
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import logging
from typing import Dict, Tuple
from datetime import datetime
import json

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.ml_recommender_v4 import ContinuousRecommenderV4
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class HairstyleDatasetV4(Dataset):
    """연속형 변수 기반 헤어스타일 데이터셋"""

    def __init__(
        self,
        face_features: np.ndarray,
        skin_features: np.ndarray,
        style_embeddings: np.ndarray,
        scores: np.ndarray
    ):
        """
        Args:
            face_features: [N, 6]
            skin_features: [N, 2]
            style_embeddings: [N, 384]
            scores: [N]
        """
        self.face_features = torch.tensor(face_features, dtype=torch.float32)
        self.skin_features = torch.tensor(skin_features, dtype=torch.float32)
        self.style_embeddings = torch.tensor(style_embeddings, dtype=torch.float32)
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
    """
    NPZ 데이터 로드 및 전처리

    Args:
        data_path: NPZ 파일 경로

    Returns:
        전처리된 데이터 딕셔너리
    """
    logger.info(f"📂 데이터 로딩: {data_path}")

    data = np.load(data_path, allow_pickle=True)

    face_features = data['face_features']  # [N, 6]
    skin_features = data['skin_features']  # [N, 2]
    hairstyles = data['hairstyles']  # [N]
    scores = data['scores']  # [N]

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

    # 데이터 정규화 (옵션)
    # face_features의 스케일이 크게 다르므로 정규화 고려
    # 여기서는 일단 원본 사용 (모델에서 BatchNorm 사용)

    # 통계
    logger.info(f"\n📊 데이터 통계:")
    logger.info(f"  - 점수 범위: {scores.min():.1f} ~ {scores.max():.1f}")
    logger.info(f"  - 점수 평균: {scores.mean():.1f} ± {scores.std():.1f}")
    logger.info(f"  - 고점수 (≥80): {(scores >= 80).sum():,}개 ({(scores >= 80).sum()/len(scores)*100:.1f}%)")
    logger.info(f"  - 저점수 (<40): {(scores < 40).sum():,}개 ({(scores < 40).sum()/len(scores)*100:.1f}%)")

    return {
        'face_features': face_features,
        'skin_features': skin_features,
        'style_embeddings': style_embeddings,
        'scores': scores,
        'hairstyles': hairstyles
    }


def create_dataloaders(
    data: Dict,
    batch_size: int = 64,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Train/Val/Test 데이터로더 생성

    Args:
        data: 전처리된 데이터
        batch_size: 배치 크기
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 데이터셋 생성
    dataset = HairstyleDatasetV4(
        face_features=data['face_features'],
        skin_features=data['skin_features'],
        style_embeddings=data['style_embeddings'],
        scores=data['scores']
    )

    # Train/Val/Test 분할
    total_size = len(dataset)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"\n📊 데이터 분할:")
    logger.info(f"  - Train: {len(train_dataset):,} ({len(train_dataset)/total_size*100:.1f}%)")
    logger.info(f"  - Val:   {len(val_dataset):,} ({len(val_dataset)/total_size*100:.1f}%)")
    logger.info(f"  - Test:  {len(test_dataset):,} ({len(test_dataset)/total_size*100:.1f}%)")

    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows에서는 0 권장
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

        # Forward
        optimizer.zero_grad()
        pred_scores = model(face_feat, skin_feat, style_emb)

        # Loss
        loss = criterion(pred_scores, scores)

        # Backward
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
) -> Tuple[float, float]:
    """검증"""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0

    with torch.no_grad():
        for face_feat, skin_feat, style_emb, scores in val_loader:
            face_feat = face_feat.to(device)
            skin_feat = skin_feat.to(device)
            style_emb = style_emb.to(device)
            scores = scores.to(device)

            # Predict
            pred_scores = model(face_feat, skin_feat, style_emb)

            # Loss
            loss = criterion(pred_scores, scores)
            mae = torch.abs(pred_scores - scores).mean()

            total_loss += loss.item()
            total_mae += mae.item()

    avg_loss = total_loss / len(val_loader)
    avg_mae = total_mae / len(val_loader)

    return avg_loss, avg_mae


def train_model(
    data_path: str,
    output_dir: str = "models",
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 15,
    use_attention: bool = True
):
    """
    모델 학습 메인 함수

    Args:
        data_path: NPZ 데이터 경로
        output_dir: 모델 저장 디렉토리
        epochs: 학습 에폭 수
        batch_size: 배치 크기
        learning_rate: 학습률
        patience: Early stopping patience
        use_attention: Attention 사용 여부
    """
    logger.info("=" * 60)
    logger.info("🚀 v4 모델 학습 시작")
    logger.info("=" * 60)

    # 출력 디렉토리 생성
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  디바이스: {device}")

    # 데이터 로드
    data = load_training_data(data_path)

    # 데이터로더 생성
    train_loader, val_loader, test_loader = create_dataloaders(
        data,
        batch_size=batch_size
    )

    # 모델 생성
    model = ContinuousRecommenderV4(
        face_feat_dim=6,
        skin_feat_dim=2,
        style_embed_dim=384,
        use_attention=use_attention,
        dropout_rate=0.3
    ).to(device)

    logger.info(f"\n🏗️  모델 구조:")
    logger.info(f"  - Face features: 6 → 64")
    logger.info(f"  - Skin features: 2 → 32")
    logger.info(f"  - Style embedding: 384")
    logger.info(f"  - Total input: 480")
    logger.info(f"  - Attention: {use_attention}")

    # 학습 설정
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # 학습 기록
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'lr': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    # 학습 시작
    logger.info(f"\n🏋️  학습 시작:")
    logger.info(f"  - Epochs: {epochs}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Learning rate: {learning_rate}")
    logger.info(f"  - Early stopping patience: {patience}")

    for epoch in range(epochs):
        # 학습
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # 검증
        val_loss, val_mae = validate(model, val_loader, criterion, device)

        # 학습률
        current_lr = optimizer.param_groups[0]['lr']

        # 기록
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['lr'].append(current_lr)

        # 출력
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch+1:3d}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MAE: {val_mae:.2f} | "
                f"LR: {current_lr:.6f}"
            )

        # 스케줄러 업데이트
        scheduler.step(val_loss)

        # Best 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            model_path = output_dir / "hairstyle_recommender_v4.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history,
                'config': {
                    'face_feat_dim': 6,
                    'skin_feat_dim': 2,
                    'style_embed_dim': 384,
                    'use_attention': use_attention
                }
            }, model_path)

            logger.info(f"  ✅ Best 모델 저장: {model_path}")

        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"\n⏸️  Early stopping at epoch {epoch + 1}")
            break

    # 최종 테스트
    logger.info(f"\n🧪 최종 테스트 평가:")
    test_loss, test_mae = validate(model, test_loader, criterion, device)
    logger.info(f"  - Test Loss: {test_loss:.4f}")
    logger.info(f"  - Test MAE: {test_mae:.2f}")

    # 학습 기록 저장
    history_path = output_dir / "training_history_v4.json"
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    logger.info(f"\n📊 학습 기록 저장: {history_path}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ 학습 완료!")
    logger.info("=" * 60)
    logger.info(f"  - Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"  - Test MAE: {test_mae:.2f}")
    logger.info(f"  - 모델 경로: {output_dir / 'hairstyle_recommender_v4.pt'}")


def main():
    parser = argparse.ArgumentParser(description="v4 모델 학습")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="학습 데이터 NPZ 경로"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="모델 저장 디렉토리 (기본: models)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="학습 에폭 수 (기본: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="배치 크기 (기본: 64)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="학습률 (기본: 0.001)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (기본: 15)"
    )
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help="Attention 비활성화"
    )

    args = parser.parse_args()

    train_model(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        use_attention=not args.no_attention
    )


if __name__ == "__main__":
    main()
