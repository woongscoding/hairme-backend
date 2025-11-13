#!/usr/bin/env python3
"""
사용자 피드백 기반 모델 재학습

DB에 저장된 실제 사용자 피드백으로 ML 모델을 재학습합니다.

MLOps 워크플로우:
1. DB에서 피드백 데이터 로드
2. 기존 합성 데이터와 병합
3. 모델 재학습
4. 성능 평가
5. 모델 배포

Author: HairMe ML Team
Date: 2025-11-08
Version: 1.0.0
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import json

# Windows 인코딩 문제 해결
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# SQLAlchemy imports
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ==================== 설정 ====================
class Config:
    """재학습 설정"""

    # DB 연결 (환경변수에서 로드)
    DB_URL = "sqlite:///./hairstyle.db"  # 기본값

    # 모델 경로
    MODEL_PATH = "models/hairstyle_recommender.pt"
    EMBEDDINGS_PATH = "data_source/style_embeddings.npz"
    BACKUP_DIR = "models/backups"

    # 학습 설정
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005  # 재학습은 낮은 learning rate
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # 데이터 비율
    MIN_FEEDBACK_COUNT = 10  # 최소 피드백 개수
    SYNTHETIC_WEIGHT = 0.7   # 합성 데이터 가중치
    FEEDBACK_WEIGHT = 0.3    # 피드백 데이터 가중치


# ==================== DB 모델 (간소화) ====================
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime

Base = declarative_base()


class UserFeedback(Base):
    """사용자 피드백 테이블"""
    __tablename__ = "user_feedback"

    id = Column(Integer, primary_key=True)
    face_shape = Column(String(20))
    skin_tone = Column(String(20))
    hairstyle = Column(String(100))
    reaction = Column(Integer)  # 1: 좋아요, 0: 싫어요
    ml_score = Column(Float)
    created_at = Column(DateTime)


# ==================== 데이터 로더 ====================
class FeedbackLoader:
    """DB에서 피드백 데이터 로드"""

    def __init__(self, db_url: str):
        """초기화"""
        self.engine = create_engine(db_url)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def load_feedbacks(self) -> List[Dict]:
        """
        DB에서 피드백 로드

        Returns:
            피드백 리스트
        """
        print(f"\n📂 DB에서 피드백 로딩...")

        feedbacks = self.session.query(UserFeedback).all()

        results = []
        for fb in feedbacks:
            results.append({
                "face_shape": fb.face_shape,
                "skin_tone": fb.skin_tone,
                "hairstyle": fb.hairstyle,
                "reaction": fb.reaction,  # 1 or 0
                "ml_score": fb.ml_score
            })

        print(f"  ✅ {len(results)}개 피드백 로드")

        # 통계
        likes = sum(1 for r in results if r["reaction"] == 1)
        dislikes = len(results) - likes

        print(f"  📊 좋아요: {likes}개, 싫어요: {dislikes}개")

        return results

    def close(self):
        """세션 종료"""
        self.session.close()


class DataMerger:
    """합성 데이터와 피드백 데이터 병합"""

    FACE_SHAPES = ["각진형", "둥근형", "긴형", "계란형"]
    SKIN_TONES = ["겨울쿨", "가을웜", "봄웜", "여름쿨"]

    def __init__(self, embeddings_path: str):
        """초기화"""
        # 임베딩 로드
        data = np.load(embeddings_path, allow_pickle=True)
        self.styles = data['styles'].tolist()
        self.embeddings = data['embeddings']
        self.style_to_idx = {style: idx for idx, style in enumerate(self.styles)}

    @staticmethod
    def get_adjustment_factors(feedback_count: int) -> Tuple[float, float]:
        """
        피드백 개수에 따라 조정 비율 결정

        Args:
            feedback_count: 총 피드백 개수

        Returns:
            (boost_factor, penalty_factor) - 좋아요/싫어요 조정 비율
        """
        if feedback_count < 100:
            # Phase 1: 초기 - 공격적 학습 (빠른 다양성 확보)
            return 1.2, 0.8  # 20% 변화
        elif feedback_count < 500:
            # Phase 2: 성장 - 표준 학습 (균형)
            return 1.15, 0.85  # 15% 변화
        else:
            # Phase 3: 안정 - 보수적 학습 (Fine-tuning)
            return 1.1, 0.9  # 10% 변화

    def convert_feedback_to_training_data(
        self,
        feedbacks: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        피드백을 학습 데이터로 변환

        Args:
            feedbacks: 피드백 리스트

        Returns:
            (X, y) - 특징 행렬, 타겟 벡터
        """
        print(f"\n🔄 피드백 데이터 변환 중...")

        # 피드백 개수에 따라 조정 비율 결정
        feedback_count = len(feedbacks)
        boost_factor, penalty_factor = self.get_adjustment_factors(feedback_count)

        if feedback_count < 100:
            phase = "Phase 1: 초기 (공격적)"
        elif feedback_count < 500:
            phase = "Phase 2: 성장 (표준)"
        else:
            phase = "Phase 3: 안정 (보수적)"

        print(f"  📊 {phase}")
        print(f"  📈 조정 비율: 좋아요 {boost_factor}배, 싫어요 {penalty_factor}배")

        X_list = []
        y_list = []

        for fb in feedbacks:
            # 얼굴형 one-hot
            face_vec = np.zeros(4, dtype=np.float32)
            if fb["face_shape"] in self.FACE_SHAPES:
                idx = self.FACE_SHAPES.index(fb["face_shape"])
                face_vec[idx] = 1.0

            # 피부톤 one-hot
            tone_vec = np.zeros(4, dtype=np.float32)
            if fb["skin_tone"] in self.SKIN_TONES:
                idx = self.SKIN_TONES.index(fb["skin_tone"])
                tone_vec[idx] = 1.0

            # 헤어스타일 임베딩
            hairstyle = fb["hairstyle"]
            if hairstyle not in self.style_to_idx:
                continue  # 미등록 스타일 스킵

            style_idx = self.style_to_idx[hairstyle]
            style_vec = self.embeddings[style_idx]

            # 특징 벡터 생성
            feature = np.concatenate([face_vec, tone_vec, style_vec])
            X_list.append(feature)

            # 타겟 점수 생성 (동적 비례 조정)
            # 피드백 개수에 따라 조정 강도 자동 변화
            ml_score = fb["ml_score"]
            if fb["reaction"] == 1:
                score = ml_score * boost_factor  # 좋아요
            else:
                score = ml_score * penalty_factor  # 싫어요

            # 0-100 범위 제한
            score = max(0.0, min(100.0, score))

            y_list.append(score)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        print(f"  ✅ 변환 완료: {len(X)}개 샘플")
        print(f"  📊 점수 범위: {y.min():.1f} ~ {y.max():.1f}")

        return X, y

    def merge_with_synthetic(
        self,
        feedback_X: np.ndarray,
        feedback_y: np.ndarray,
        synthetic_path: str,
        synthetic_weight: float = 0.7,
        feedback_weight: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        피드백 데이터와 합성 데이터 병합

        Args:
            feedback_X, feedback_y: 피드백 데이터
            synthetic_path: 합성 데이터 경로
            synthetic_weight: 합성 데이터 가중치
            feedback_weight: 피드백 데이터 가중치

        Returns:
            (X_merged, y_merged)
        """
        print(f"\n🔗 합성 데이터와 병합 중...")

        # 합성 데이터 로드
        synthetic_data = np.load(synthetic_path)
        synthetic_X_train = synthetic_data['X_train']
        synthetic_y_train = synthetic_data['y_train']

        print(f"  합성 데이터: {len(synthetic_X_train)}개")
        print(f"  피드백 데이터: {len(feedback_X)}개")

        # 샘플 수 조정 (가중치 적용)
        synthetic_count = int(len(synthetic_X_train) * synthetic_weight / (synthetic_weight + feedback_weight))
        feedback_count = int(len(feedback_X) * feedback_weight / (synthetic_weight + feedback_weight))

        # 샘플링
        synthetic_indices = np.random.choice(len(synthetic_X_train), synthetic_count, replace=False)
        feedback_indices = np.random.choice(len(feedback_X), min(feedback_count, len(feedback_X)), replace=True)

        X_merged = np.vstack([
            synthetic_X_train[synthetic_indices],
            feedback_X[feedback_indices]
        ])

        y_merged = np.concatenate([
            synthetic_y_train[synthetic_indices],
            feedback_y[feedback_indices]
        ])

        print(f"  ✅ 병합 완료: {len(X_merged)}개 (합성 {synthetic_count} + 피드백 {len(feedback_indices)})")

        return X_merged, y_merged


# ==================== 모델 및 학습 (기존 코드 재사용) ====================
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


def retrain_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_path: str,
    num_epochs: int = 50
):
    """모델 재학습"""

    print(f"\n🚀 모델 재학습 시작...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 기존 모델 로드
    model = RecommendationModel()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
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

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\n  ⚠️  Early Stopping at epoch {epoch+1}")
            break

    print(f"\n  ✅ 재학습 완료! 최종 Val Loss: {val_loss:.4f}")

    return model


# ==================== 메인 함수 ====================
def main():
    """메인 실행"""

    parser = argparse.ArgumentParser(description="사용자 피드백 기반 모델 재학습")
    parser.add_argument('--db-url', type=str, default=Config.DB_URL, help='DB URL')
    parser.add_argument('--min-feedbacks', type=int, default=Config.MIN_FEEDBACK_COUNT, help='최소 피드백 개수')

    args = parser.parse_args()

    print("=" * 60)
    print("🔄 MLOps 재학습 파이프라인 v1.0.0")
    print("=" * 60)

    try:
        # 1. 피드백 로드
        loader = FeedbackLoader(args.db_url)
        feedbacks = loader.load_feedbacks()
        loader.close()

        if len(feedbacks) < args.min_feedbacks:
            print(f"\n⚠️  피드백이 부족합니다 ({len(feedbacks)}개 < {args.min_feedbacks}개)")
            print("   더 많은 피드백을 수집한 후 재학습하세요.")
            return 1

        # 2. 피드백 데이터 변환
        merger = DataMerger(Config.EMBEDDINGS_PATH)
        feedback_X, feedback_y = merger.convert_feedback_to_training_data(feedbacks)

        # 3. 합성 데이터와 병합
        X_merged, y_merged = merger.merge_with_synthetic(
            feedback_X, feedback_y,
            "data_source/ml_training_dataset.npz",
            Config.SYNTHETIC_WEIGHT,
            Config.FEEDBACK_WEIGHT
        )

        # 4. Train/Val split
        X_train, X_val, y_train, y_val = train_test_split(
            X_merged, y_merged,
            test_size=0.2,
            random_state=42
        )

        print(f"\n📊 최종 데이터:")
        print(f"  - Train: {len(X_train)}개")
        print(f"  - Val: {len(X_val)}개")

        # 5. 기존 모델 백업
        backup_dir = Path(Config.BACKUP_DIR)
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"model_backup_{timestamp}.pt"

        import shutil
        shutil.copy(Config.MODEL_PATH, backup_path)
        print(f"\n💾 기존 모델 백업: {backup_path}")

        # 6. 재학습
        new_model = retrain_model(X_train, y_train, X_val, y_val, Config.MODEL_PATH, Config.NUM_EPOCHS)

        # 7. 새 모델 저장
        torch.save(new_model.state_dict(), Config.MODEL_PATH)
        print(f"✅ 새 모델 저장: {Config.MODEL_PATH}")

        print("\n" + "=" * 60)
        print("🎉 재학습 완료!")
        print("=" * 60)
        print(f"📊 결과:")
        print(f"  - 피드백 데이터: {len(feedbacks)}개")
        print(f"  - 최종 학습 데이터: {len(X_train)}개")
        print(f"  - 백업 파일: {backup_path.name}")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
