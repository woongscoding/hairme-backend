#!/usr/bin/env python3
"""
헤어스타일 추천 모델 학습

PyTorch로 Neural Network 회귀 모델 학습

Model Architecture:
  Input (392) → Dense(256) → ReLU → Dropout(0.3)
              → Dense(128) → ReLU → Dropout(0.2)
              → Dense(64) → ReLU
              → Dense(1) - Output

Loss: MSE (Mean Squared Error)
Optimizer: Adam

Author: HairMe ML Team
Date: 2025-11-08
Version: 1.0.0
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt

# Windows 인코딩 문제 해결
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter


# ==================== 설정 ====================
class Config:
    """학습 설정"""

    # 모델 구조
    INPUT_DIM = 392
    HIDDEN_DIMS = [256, 128, 64]
    OUTPUT_DIM = 1
    DROPOUT_RATES = [0.3, 0.2, 0.0]

    # 학습 하이퍼파라미터
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15

    # 입력/출력 경로
    DEFAULT_DATASET_PATH = "data_source/ml_training_dataset.npz"
    DEFAULT_OUTPUT_DIR = "models"


# ==================== 데이터셋 ====================
class HairstyleDataset(Dataset):
    """PyTorch Dataset for hairstyle recommendation"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        초기화

        Args:
            X: 특징 행렬 (N, 392)
            y: 타겟 벡터 (N,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)  # (N, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ==================== 모델 ====================
class RecommendationModel(nn.Module):
    """Neural Network for hairstyle recommendation score prediction"""

    def __init__(
        self,
        input_dim: int = Config.INPUT_DIM,
        hidden_dims: list = None,
        dropout_rates: list = None
    ):
        """
        초기화

        Args:
            input_dim: 입력 차원
            hidden_dims: 은닉층 차원 리스트
            dropout_rates: 드롭아웃 비율 리스트
        """
        super(RecommendationModel, self).__init__()

        if hidden_dims is None:
            hidden_dims = Config.HIDDEN_DIMS
        if dropout_rates is None:
            dropout_rates = Config.DROPOUT_RATES

        layers = []
        prev_dim = input_dim

        # 은닉층 구축
        for hidden_dim, dropout_rate in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # 출력층
        layers.append(nn.Linear(prev_dim, Config.OUTPUT_DIM))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        return self.network(x)


# ==================== 학습 ====================
class Trainer:
    """모델 학습"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device
    ):
        """
        초기화

        Args:
            model: 학습할 모델
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            device: 학습 디바이스
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

        # 학습 이력
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }

        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self) -> Tuple[float, float]:
        """1 에폭 학습"""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        n_batches = 0

        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)

            # Backward
            loss.backward()
            self.optimizer.step()

            # 통계
            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(predictions - y_batch)).item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_mae = total_mae / n_batches

        return avg_loss, avg_mae

    def validate(self) -> Tuple[float, float]:
        """검증"""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        n_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)

                total_loss += loss.item()
                total_mae += torch.mean(torch.abs(predictions - y_batch)).item()
                n_batches += 1

        avg_loss = total_loss / n_batches
        avg_mae = total_mae / n_batches

        return avg_loss, avg_mae

    def train(self, num_epochs: int = Config.NUM_EPOCHS) -> Dict:
        """전체 학습 루프"""
        print(f"\n🚀 학습 시작...")
        print(f"  에폭 수: {num_epochs}")
        print(f"  배치 크기: {Config.BATCH_SIZE}")
        print(f"  학습률: {Config.LEARNING_RATE}")
        print(f"  Early Stopping Patience: {Config.EARLY_STOPPING_PATIENCE}")
        print(f"  디바이스: {self.device}")

        start_time = time.time()

        for epoch in range(num_epochs):
            # 학습
            train_loss, train_mae = self.train_epoch()
            val_loss, val_mae = self.validate()

            # 이력 저장
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)

            # 출력 (10 에폭마다)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.2f} | "
                      f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.2f}")

            # Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\n  ⚠️  Early Stopping at epoch {epoch+1}")
                break

        elapsed_time = time.time() - start_time

        print(f"\n  ✅ 학습 완료!")
        print(f"  ⏱️  소요 시간: {elapsed_time:.1f}초")
        print(f"  📊 최종 성능:")
        print(f"    - Train Loss: {self.history['train_loss'][-1]:.4f}, "
              f"MAE: {self.history['train_mae'][-1]:.2f}")
        print(f"    - Val Loss: {self.history['val_loss'][-1]:.4f}, "
              f"MAE: {self.history['val_mae'][-1]:.2f}")

        return self.history


# ==================== 클래스 가중치 ====================
def compute_sample_weights(X: np.ndarray, skin_tone_names: list = None) -> np.ndarray:
    """
    피부톤 불균형 해결을 위한 샘플 가중치 계산

    Args:
        X: 특징 행렬 (N, 392) - [face(4) + tone(4) + emb(384)]
        skin_tone_names: 피부톤 이름 리스트 (출력용)

    Returns:
        각 샘플의 가중치 배열 (N,)
    """
    if skin_tone_names is None:
        skin_tone_names = ["겨울쿨", "가을웜", "봄웜", "여름쿨"]

    print(f"\n⚖️  피부톤 클래스 가중치 계산 중...")

    # 피부톤 one-hot은 인덱스 4-7
    skin_tone_onehot = X[:, 4:8]

    # 각 샘플의 피부톤 클래스 추출 (argmax)
    skin_tone_classes = np.argmax(skin_tone_onehot, axis=1)

    # 클래스별 샘플 수
    class_counts = Counter(skin_tone_classes)

    print(f"\n  📊 피부톤 분포:")
    for class_idx, count in sorted(class_counts.items()):
        tone_name = skin_tone_names[class_idx] if class_idx < len(skin_tone_names) else f"Unknown_{class_idx}"
        percentage = count / len(X) * 100
        print(f"    - {tone_name}: {count}개 ({percentage:.1f}%)")

    # 클래스 가중치 계산 (balanced)
    # weight = n_samples / (n_classes * n_samples_per_class)
    n_samples = len(X)
    n_classes = len(class_counts)

    class_weights = {}
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = n_samples / (n_classes * count)

    print(f"\n  ⚖️  클래스 가중치:")
    for class_idx, weight in sorted(class_weights.items()):
        tone_name = skin_tone_names[class_idx] if class_idx < len(skin_tone_names) else f"Unknown_{class_idx}"
        print(f"    - {tone_name}: {weight:.3f}")

    # 각 샘플에 가중치 할당
    sample_weights = np.array([class_weights[class_idx] for class_idx in skin_tone_classes])

    print(f"\n  ✅ 샘플 가중치 계산 완료")
    print(f"    - 최소 가중치: {sample_weights.min():.3f}")
    print(f"    - 최대 가중치: {sample_weights.max():.3f}")
    print(f"    - 가중치 비율: 1:{sample_weights.max()/sample_weights.min():.1f}")

    return sample_weights


# ==================== 저장 및 시각화 ====================
class ModelExporter:
    """모델 및 결과 저장"""

    @staticmethod
    def save_model(model: nn.Module, output_dir: Path):
        """모델 저장"""
        print(f"\n💾 모델 저장 중...")

        # PyTorch 모델 저장
        model_path = output_dir / "hairstyle_recommender.pt"
        torch.save(model.state_dict(), model_path)
        print(f"  ✅ 모델 저장: {model_path}")

        # 모델 크기
        model_size = model_path.stat().st_size / 1024
        print(f"  📊 모델 크기: {model_size:.1f} KB")

    @staticmethod
    def save_history(history: Dict, output_dir: Path):
        """학습 이력 저장"""
        history_path = output_dir / "training_history.json"

        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

        print(f"  ✅ 학습 이력 저장: {history_path}")

    @staticmethod
    def plot_history(history: Dict, output_dir: Path):
        """학습 곡선 시각화"""
        print(f"\n📊 학습 곡선 저장 중...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss 곡선
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # MAE 곡선
        ax2.plot(history['train_mae'], label='Train MAE')
        ax2.plot(history['val_mae'], label='Val MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE (Mean Absolute Error)')
        ax2.set_title('Training and Validation MAE')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        plot_path = output_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✅ 학습 곡선 저장: {plot_path}")


# ==================== 메인 함수 ====================
def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="헤어스타일 추천 모델 학습"
    )

    parser.add_argument(
        '-d', '--dataset',
        type=str,
        default=Config.DEFAULT_DATASET_PATH,
        help=f'데이터셋 경로 (기본값: {Config.DEFAULT_DATASET_PATH})'
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=Config.DEFAULT_OUTPUT_DIR,
        help=f'모델 저장 경로 (기본값: {Config.DEFAULT_OUTPUT_DIR})'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=Config.NUM_EPOCHS,
        help=f'학습 에폭 수 (기본값: {Config.NUM_EPOCHS})'
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🎨 헤어스타일 추천 모델 학습 v1.0.0")
    print("=" * 60)
    print(f"데이터셋: {dataset_path}")
    print(f"모델 저장 경로: {output_dir.absolute()}")
    print("=" * 60)

    try:
        # 1. 데이터 로드
        print(f"\n📂 데이터셋 로딩...")
        data = np.load(dataset_path)
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']

        print(f"  ✅ 로드 완료:")
        print(f"    - Train: {len(X_train)}개")
        print(f"    - Val: {len(X_val)}개")

        # 2. 피부톤 클래스 가중치 계산
        sample_weights = compute_sample_weights(X_train)

        # 3. 데이터셋 및 로더 생성
        train_dataset = HairstyleDataset(X_train, y_train)
        val_dataset = HairstyleDataset(X_val, y_val)

        # WeightedRandomSampler 생성 (피부톤 불균형 해결)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True  # 중복 샘플링 허용
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            sampler=sampler  # shuffle=True 대신 sampler 사용
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False
        )

        print(f"\n  ✅ DataLoader 생성 완료:")
        print(f"    - Train Loader: WeightedRandomSampler 적용 (피부톤 균형)")
        print(f"    - Batch Size: {Config.BATCH_SIZE}")

        # 4. 모델 생성
        print(f"\n🤖 모델 생성...")
        model = RecommendationModel()

        # 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  ✅ 모델 구조:")
        print(f"    - 입력: {Config.INPUT_DIM}차원")
        print(f"    - 은닉층: {Config.HIDDEN_DIMS}")
        print(f"    - 출력: {Config.OUTPUT_DIM}차원")
        print(f"    - 총 파라미터: {total_params:,}개")
        print(f"    - 학습 가능 파라미터: {trainable_params:,}개")

        # 5. 디바이스 설정
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 6. 학습
        trainer = Trainer(model, train_loader, val_loader, device)
        history = trainer.train(num_epochs=args.epochs)

        # 7. 저장
        exporter = ModelExporter()
        exporter.save_model(model, output_dir)
        exporter.save_history(history, output_dir)
        exporter.plot_history(history, output_dir)

        print("\n" + "=" * 60)
        print("🎉 학습 완료!")
        print("=" * 60)
        print(f"📊 결과:")
        print(f"  - 최종 Val MAE: {history['val_mae'][-1]:.2f}점")
        print(f"  - 저장된 파일:")
        print(f"    * {output_dir / 'hairstyle_recommender.pt'}")
        print(f"    * {output_dir / 'training_history.json'}")
        print(f"    * {output_dir / 'training_curves.png'}")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
