#!/usr/bin/env python3
"""
사용자 피드백 기반 모델 재학습 (v2.0)

Progressive Data Mixing 전략:
- 피드백 < 500개: 100% Gemini, 0% Feedback
- 피드백 < 1000개: 70% Gemini, 30% Feedback
- 피드백 < 2000개: 40% Gemini, 60% Feedback
- 피드백 < 5000개: 20% Gemini, 80% Feedback
- 피드백 >= 5000개: 0% Gemini, 100% Feedback

Ground Truth Rules:
- 👍 (like) → 90.0 (user LIKED this combination)
- 👎 (dislike) → 10.0 (user DISLIKED this combination)

Author: HairMe ML Team
Date: 2025-11-13
Version: 2.0.0
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple
import numpy as np
import json
from datetime import datetime

# Windows 인코딩 문제 해결
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ==================== 설정 ====================
class Config:
    """재학습 설정"""

    # 데이터 경로
    FEEDBACK_NPZ_PATH = "data/feedback_training_data.npz"
    GEMINI_NPZ_PATH = "data_source/ml_training_dataset.npz"
    MODEL_PATH = "models/hairstyle_recommender.pt"
    BACKUP_DIR = "models/backups"

    # 학습 설정
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001  # 재학습은 낮은 learning rate
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15

    # 디바이스
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== Progressive Mixing ====================
def get_mixing_ratios(feedback_count: int) -> Tuple[float, float]:
    """
    피드백 개수에 따라 Gemini/Feedback 데이터 혼합 비율 결정

    Args:
        feedback_count: 총 피드백 개수

    Returns:
        (gemini_ratio, feedback_ratio)
    """
    if feedback_count < 500:
        phase = "Phase 1: Bootstrapping"
        gemini_ratio, feedback_ratio = 1.0, 0.0
    elif feedback_count < 1000:
        phase = "Phase 2: Initial Learning"
        gemini_ratio, feedback_ratio = 0.7, 0.3
    elif feedback_count < 2000:
        phase = "Phase 3: Balanced Learning"
        gemini_ratio, feedback_ratio = 0.4, 0.6
    elif feedback_count < 5000:
        phase = "Phase 4: User-Driven Learning"
        gemini_ratio, feedback_ratio = 0.2, 0.8
    else:
        phase = "Phase 5: Pure Feedback"
        gemini_ratio, feedback_ratio = 0.0, 1.0

    print(f"\n📊 {phase}")
    print(f"  - Gemini 데이터: {gemini_ratio*100:.0f}%")
    print(f"  - Feedback 데이터: {feedback_ratio*100:.0f}%")

    return gemini_ratio, feedback_ratio


def prepare_training_data(feedback_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    학습 데이터 준비 (Progressive Mixing)

    Args:
        feedback_count: 피드백 개수 (이미 알고 있는 값)

    Returns:
        (X, y) - 특징 행렬, 타겟 벡터
    """
    print("\n" + "="*60)
    print("📂 학습 데이터 준비 중...")
    print("="*60)

    # 1. Mixing 비율 결정
    gemini_ratio, feedback_ratio = get_mixing_ratios(feedback_count)

    X_list = []
    y_list = []

    # 2. Gemini 데이터 로드 (필요 시)
    if gemini_ratio > 0:
        try:
            gemini_data = np.load(Config.GEMINI_NPZ_PATH)
            gemini_X = gemini_data['X_train']
            gemini_y = gemini_data['y_train']

            # Gemini 데이터 샘플링
            gemini_sample_size = int(len(gemini_X) * gemini_ratio)
            gemini_indices = np.random.choice(
                len(gemini_X),
                gemini_sample_size,
                replace=False
            )

            X_list.append(gemini_X[gemini_indices])
            y_list.append(gemini_y[gemini_indices])

            print(f"\n✅ Gemini 데이터 로드: {len(gemini_X)}개 → {gemini_sample_size}개 샘플링")

        except FileNotFoundError:
            print(f"\n⚠️  Gemini 데이터 파일 없음: {Config.GEMINI_NPZ_PATH}")

    # 3. Feedback 데이터 로드 (필요 시)
    if feedback_ratio > 0:
        try:
            feedback_data = np.load(Config.FEEDBACK_NPZ_PATH)
            feedback_X = feedback_data['X']
            feedback_y = feedback_data['y']

            X_list.append(feedback_X)
            y_list.append(feedback_y)

            print(f"✅ Feedback 데이터 로드: {len(feedback_X)}개")

        except FileNotFoundError:
            print(f"\n⚠️  Feedback 데이터 파일 없음: {Config.FEEDBACK_NPZ_PATH}")

    # 4. 데이터 병합
    if not X_list:
        raise ValueError("학습 데이터가 없습니다!")

    X_combined = np.vstack(X_list)
    y_combined = np.concatenate(y_list)

    print(f"\n✅ 최종 데이터: {len(X_combined)}개")
    print(f"  - 특징 차원: {X_combined.shape[1]}")
    print(f"  - 점수 범위: {y_combined.min():.1f} ~ {y_combined.max():.1f}")

    return X_combined, y_combined


# ==================== 모델 정의 ====================
class RecommendationModel(nn.Module):
    """PyTorch 추천 모델"""

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


class HairstyleDataset(Dataset):
    """PyTorch Dataset"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ==================== 학습 함수 ====================
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_path: str,
    num_epochs: int = 100
) -> nn.Module:
    """
    모델 재학습

    Args:
        X_train, y_train: 학습 데이터
        X_val, y_val: 검증 데이터
        model_path: 기존 모델 경로
        num_epochs: 에폭 수

    Returns:
        학습된 모델
    """
    print("\n" + "="*60)
    print("🚀 모델 재학습 시작")
    print("="*60)

    device = Config.DEVICE
    print(f"디바이스: {device}")

    # 기존 모델 로드
    model = RecommendationModel()

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"✅ 기존 모델 로드: {model_path}")
    except FileNotFoundError:
        print(f"⚠️  기존 모델 없음, 새로 초기화")

    model.to(device)

    # 데이터셋
    train_dataset = HairstyleDataset(X_train, y_train)
    val_dataset = HairstyleDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 옵티마이저
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\n학습 설정:")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch Size: {Config.BATCH_SIZE}")
    print(f"  - Learning Rate: {Config.LEARNING_RATE}")
    print(f"  - Early Stopping Patience: {Config.EARLY_STOPPING_PATIENCE}")

    print("\n학습 시작...\n")

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # 진행 상황 출력
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\n⚠️  Early Stopping at epoch {epoch+1}")
            break

    print(f"\n✅ 재학습 완료! 최종 Val Loss: {val_loss:.4f}")

    return model


# ==================== 메인 함수 ====================
def main():
    """메인 실행"""

    parser = argparse.ArgumentParser(
        description="사용자 피드백 기반 모델 재학습 (Progressive Mixing v2.0)"
    )
    parser.add_argument(
        '--feedback-count',
        type=int,
        required=True,
        help='현재 피드백 개수 (500, 1000, 2000, 5000)'
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='학습 후 모델 저장'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=Config.NUM_EPOCHS,
        help=f'에폭 수 (기본값: {Config.NUM_EPOCHS})'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🔄 MLOps 재학습 파이프라인 v2.0")
    print("   (Progressive Data Mixing Strategy)")
    print("=" * 60)
    print(f"피드백 개수: {args.feedback_count}")
    print(f"모델 저장: {'예' if args.save_model else '아니오'}")
    print("=" * 60)

    try:
        # 1. 학습 데이터 준비 (Progressive Mixing)
        X_combined, y_combined = prepare_training_data(args.feedback_count)

        # 2. Train/Val Split
        X_train, X_val, y_train, y_val = train_test_split(
            X_combined, y_combined,
            test_size=0.2,
            random_state=42
        )

        print(f"\n📊 데이터 분할:")
        print(f"  - Train: {len(X_train)}개")
        print(f"  - Val: {len(X_val)}개")

        # 3. 기존 모델 백업 (저장 모드일 때만)
        if args.save_model:
            backup_dir = Path(Config.BACKUP_DIR)
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            feedback_count = args.feedback_count
            backup_filename = f"model_backup_{feedback_count}_{timestamp}.pt"
            backup_path = backup_dir / backup_filename

            try:
                import shutil
                shutil.copy(Config.MODEL_PATH, backup_path)
                print(f"\n💾 기존 모델 백업: {backup_path}")
            except FileNotFoundError:
                print(f"\n⚠️  기존 모델 없음, 백업 생략")

        # 4. 재학습
        new_model = train_model(
            X_train, y_train,
            X_val, y_val,
            Config.MODEL_PATH,
            args.epochs
        )

        # 5. 새 모델 저장
        if args.save_model:
            new_model_path = Config.MODEL_PATH.replace(
                '.pt',
                f'_v2_{args.feedback_count}.pt'
            )
            torch.save(new_model.state_dict(), new_model_path)
            print(f"\n✅ 새 모델 저장: {new_model_path}")

            # 메타데이터 저장
            metadata = {
                "version": "2.0",
                "feedback_count": args.feedback_count,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "timestamp": datetime.now().isoformat()
            }

            metadata_path = new_model_path.replace('.pt', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            print(f"✅ 메타데이터 저장: {metadata_path}")

        print("\n" + "=" * 60)
        print("🎉 재학습 완료!")
        print("=" * 60)
        print(f"📊 결과:")
        print(f"  - 피드백 개수: {args.feedback_count}")
        print(f"  - 학습 데이터: {len(X_train)}개")
        print(f"  - 검증 데이터: {len(X_val)}개")
        if args.save_model:
            print(f"  - 저장 경로: {new_model_path}")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
