"""
자동 모델 재학습 스크립트

- 새로운 학습 데이터로 모델 재학습
- 버전 관리 및 체크포인트 저장
- 성능 메트릭 로깅
- 기존 모델과 성능 비교
"""

import os
import sys
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    mean_squared_error,
    confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경용
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime


# 프로젝트 루트 디렉토리를 Python path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ==================== 데이터셋 클래스 ====================
class HairstyleDataset(Dataset):
    """헤어스타일 추천 데이터셋"""

    def __init__(self, csv_path, face_encoder=None, skin_encoder=None,
                 style_encoder=None, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.is_train = is_train

        if is_train:
            self.face_encoder = LabelEncoder()
            self.skin_encoder = LabelEncoder()
            self.style_encoder = LabelEncoder()

            self.face_encoded = self.face_encoder.fit_transform(self.df['face_shape'])
            self.skin_encoded = self.skin_encoder.fit_transform(self.df['skin_tone'])
            self.style_encoded = self.style_encoder.fit_transform(self.df['hairstyle'])
        else:
            self.face_encoder = face_encoder
            self.skin_encoder = skin_encoder
            self.style_encoder = style_encoder

            self.face_encoded = self.face_encoder.transform(self.df['face_shape'])
            self.skin_encoded = self.skin_encoder.transform(self.df['skin_tone'])
            self.style_encoded = self.style_encoder.transform(self.df['hairstyle'])

        self.scores = self.df['score'].values
        self.feedbacks = (self.df['feedback'] == 'like').astype(int).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'face': torch.tensor(self.face_encoded[idx], dtype=torch.long),
            'skin': torch.tensor(self.skin_encoded[idx], dtype=torch.long),
            'style': torch.tensor(self.style_encoded[idx], dtype=torch.long),
            'score': torch.tensor(self.scores[idx], dtype=torch.float32),
            'feedback': torch.tensor(self.feedbacks[idx], dtype=torch.long)
        }


# ==================== 모델 정의 ====================
class HairstyleRecommender(nn.Module):
    """헤어스타일 추천 모델"""

    def __init__(self, n_faces=5, n_skins=3, n_styles=6,
                 emb_dim=16, hidden_dim=64):
        super().__init__()

        self.face_emb = nn.Embedding(n_faces, emb_dim)
        self.skin_emb = nn.Embedding(n_skins, emb_dim)
        self.style_emb = nn.Embedding(n_styles, emb_dim)

        self.shared_layers = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.feedback_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, face, skin, style):
        face_emb = self.face_emb(face)
        skin_emb = self.skin_emb(skin)
        style_emb = self.style_emb(style)

        x = torch.cat([face_emb, skin_emb, style_emb], dim=1)
        shared = self.shared_layers(x)

        score = self.score_head(shared).squeeze()
        feedback_logits = self.feedback_head(shared)

        return score, feedback_logits


# ==================== 트레이너 클래스 ====================
class ModelTrainer:
    """모델 학습 클래스"""

    def __init__(
        self,
        train_data_path="data_source/train_data.csv",
        val_data_path="data_source/val_data.csv",
        test_data_path="data_source/test_data.csv",
        output_dir="models/checkpoints",
        device="cpu"
    ):
        self.project_root = project_root
        self.train_path = self.project_root / train_data_path
        self.val_path = self.project_root / val_data_path
        self.test_path = self.project_root / test_data_path
        self.output_dir = self.project_root / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device)
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # 학습 히스토리
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }

        # 버전 정보
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_data(self):
        """데이터 로드"""
        print("=" * 60)
        print("📂 데이터 로드 중...")
        print("=" * 60)

        # 학습 데이터 로드 (인코더 생성)
        self.train_dataset = HairstyleDataset(
            self.train_path,
            is_train=True
        )
        print(f"✅ 학습 데이터: {len(self.train_dataset)}건")

        # 검증/테스트 데이터 로드 (인코더 재사용)
        self.val_dataset = HairstyleDataset(
            self.val_path,
            face_encoder=self.train_dataset.face_encoder,
            skin_encoder=self.train_dataset.skin_encoder,
            style_encoder=self.train_dataset.style_encoder,
            is_train=False
        )
        print(f"✅ 검증 데이터: {len(self.val_dataset)}건")

        self.test_dataset = HairstyleDataset(
            self.test_path,
            face_encoder=self.train_dataset.face_encoder,
            skin_encoder=self.train_dataset.skin_encoder,
            style_encoder=self.train_dataset.style_encoder,
            is_train=False
        )
        print(f"✅ 테스트 데이터: {len(self.test_dataset)}건")

        # 인코더 저장
        self._save_encoders()

    def _save_encoders(self):
        """인코더 저장"""
        encoders = {
            'face': self.train_dataset.face_encoder,
            'skin': self.train_dataset.skin_encoder,
            'style': self.train_dataset.style_encoder
        }

        encoder_path = self.output_dir / f"encoders_{self.version}.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(encoders, f)

        # 최신 버전도 저장
        encoder_latest = self.output_dir / "encoders_latest.pkl"
        with open(encoder_latest, 'wb') as f:
            pickle.dump(encoders, f)

        print(f"✅ 인코더 저장: {encoder_path}")

    def build_model(self):
        """모델 생성"""
        print("\n" + "=" * 60)
        print("🏗️ 모델 생성 중...")
        print("=" * 60)

        n_faces = len(self.train_dataset.face_encoder.classes_)
        n_skins = len(self.train_dataset.skin_encoder.classes_)
        n_styles = len(self.train_dataset.style_encoder.classes_)

        self.model = HairstyleRecommender(
            n_faces=n_faces,
            n_skins=n_skins,
            n_styles=n_styles,
            emb_dim=16,
            hidden_dim=64
        ).to(self.device)

        print(f"✅ 모델 생성 완료")
        print(f"   얼굴형: {n_faces}개")
        print(f"   피부톤: {n_skins}개")
        print(f"   헤어스타일: {n_styles}개")

    def calculate_class_weights(self):
        """클래스 가중치 계산 (불균형 해결)"""
        feedbacks = self.train_dataset.feedbacks
        n_total = len(feedbacks)
        n_like = np.sum(feedbacks == 1)
        n_dislike = np.sum(feedbacks == 0)

        weight_like = n_total / (2 * n_like) if n_like > 0 else 1.0
        weight_dislike = n_total / (2 * n_dislike) if n_dislike > 0 else 1.0

        class_weights = torch.tensor([weight_dislike, weight_like], dtype=torch.float32)

        print(f"\n📊 클래스 불균형 처리:")
        print(f"   Like: {n_like}건 (가중치: {weight_like:.3f})")
        print(f"   Dislike: {n_dislike}건 (가중치: {weight_dislike:.3f})")

        return class_weights

    def train(self, batch_size=64, max_epochs=50, learning_rate=0.001, patience=7):
        """모델 학습"""
        print("\n" + "=" * 60)
        print("🚀 모델 학습 시작")
        print("=" * 60)

        # 데이터 로더
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size)

        # 클래스 가중치 계산
        class_weights = self.calculate_class_weights().to(self.device)

        # 손실 함수 및 옵티마이저
        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(max_epochs):
            # 학습
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                face = batch['face'].to(self.device)
                skin = batch['skin'].to(self.device)
                style = batch['style'].to(self.device)
                score_target = batch['score'].to(self.device)
                feedback_target = batch['feedback'].to(self.device)

                optimizer.zero_grad()

                score_pred, feedback_logits = self.model(face, skin, style)

                loss_score = mse_loss(score_pred, score_target)
                loss_feedback = ce_loss(feedback_logits, feedback_target)
                loss = loss_score + 2.0 * loss_feedback

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 검증
            val_loss, val_metrics = self.evaluate(val_loader, mse_loss, ce_loss)

            # 히스토리 저장
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])

            print(f"Epoch {epoch+1}/{max_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                print("   ✅ 새로운 최고 모델!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⏹️ Early stopping (patience={patience})")
                    break

        # 최고 모델 복원
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        print("\n✅ 학습 완료!")

    def evaluate(self, data_loader, mse_loss, ce_loss):
        """모델 평가"""
        self.model.eval()
        total_loss = 0.0

        all_feedback_preds = []
        all_feedback_targets = []

        with torch.no_grad():
            for batch in data_loader:
                face = batch['face'].to(self.device)
                skin = batch['skin'].to(self.device)
                style = batch['style'].to(self.device)
                score_target = batch['score'].to(self.device)
                feedback_target = batch['feedback'].to(self.device)

                score_pred, feedback_logits = self.model(face, skin, style)

                loss_score = mse_loss(score_pred, score_target)
                loss_feedback = ce_loss(feedback_logits, feedback_target)
                loss = loss_score + 2.0 * loss_feedback

                total_loss += loss.item()

                feedback_pred = torch.argmax(feedback_logits, dim=1)
                all_feedback_preds.extend(feedback_pred.cpu().numpy())
                all_feedback_targets.extend(feedback_target.cpu().numpy())

        total_loss /= len(data_loader)

        # 메트릭 계산
        accuracy = accuracy_score(all_feedback_targets, all_feedback_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_feedback_targets,
            all_feedback_preds,
            average='weighted',
            zero_division=0
        )

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        return total_loss, metrics

    def test(self):
        """테스트 세트 평가"""
        print("\n" + "=" * 60)
        print("🧪 테스트 세트 평가")
        print("=" * 60)

        test_loader = DataLoader(self.test_dataset, batch_size=64)

        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss()

        test_loss, test_metrics = self.evaluate(test_loader, mse_loss, ce_loss)

        print(f"✅ 테스트 결과:")
        print(f"   Loss: {test_loss:.4f}")
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   Precision: {test_metrics['precision']:.4f}")
        print(f"   Recall: {test_metrics['recall']:.4f}")
        print(f"   F1-Score: {test_metrics['f1']:.4f}")

        return test_metrics

    def save_model(self):
        """모델 저장"""
        print("\n" + "=" * 60)
        print("💾 모델 저장 중...")
        print("=" * 60)

        # 버전별 저장
        model_path = self.output_dir / f"model_{self.version}.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"✅ 모델 저장: {model_path}")

        # 최신 버전 저장
        model_latest = self.output_dir / "model_latest.pth"
        torch.save(self.model.state_dict(), model_latest)
        print(f"✅ 최신 모델 저장: {model_latest}")

        # 메트릭 저장
        metrics_path = self.output_dir / f"metrics_{self.version}.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        print(f"✅ 메트릭 저장: {metrics_path}")

    def plot_training_curves(self):
        """학습 곡선 플롯"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy
        axes[0, 1].plot(self.history['val_accuracy'], label='Val Accuracy', color='green')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # F1-Score
        axes[1, 0].plot(self.history['val_f1'], label='Val F1-Score', color='orange')
        axes[1, 0].set_title('Validation F1-Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Summary
        axes[1, 1].axis('off')
        summary_text = f"""
        Training Summary

        Final Train Loss: {self.history['train_loss'][-1]:.4f}
        Final Val Loss: {self.history['val_loss'][-1]:.4f}
        Best Val Accuracy: {max(self.history['val_accuracy']):.4f}
        Best Val F1: {max(self.history['val_f1']):.4f}
        Total Epochs: {len(self.history['train_loss'])}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')

        plt.tight_layout()

        # 저장
        plot_path = self.output_dir / f"training_curves_{self.version}.png"
        plt.savefig(plot_path, dpi=150)
        print(f"✅ 학습 곡선 저장: {plot_path}")

        plt.close()

    def run(self):
        """전체 프로세스 실행"""
        print("\n" + "🚀" * 30)
        print("모델 재학습 시작")
        print("🚀" * 30 + "\n")

        # 1. 데이터 로드
        self.load_data()

        # 2. 모델 생성
        self.build_model()

        # 3. 학습
        self.train()

        # 4. 테스트
        test_metrics = self.test()

        # 5. 저장
        self.save_model()
        self.plot_training_curves()

        print("\n" + "=" * 60)
        print("✅ 모델 재학습 완료!")
        print("=" * 60)

        return {
            'version': self.version,
            'test_metrics': test_metrics,
            'model_path': self.output_dir / f"model_{self.version}.pth"
        }


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="모델 재학습")
    parser.add_argument("--batch-size", type=int, default=64, help="배치 크기")
    parser.add_argument("--epochs", type=int, default=50, help="최대 에폭 수")
    parser.add_argument("--lr", type=float, default=0.001, help="학습률")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")

    args = parser.parse_args()

    try:
        trainer = ModelTrainer()
        result = trainer.run()

        print(f"\n✅ 생성된 모델:")
        print(f"   버전: {result['version']}")
        print(f"   경로: {result['model_path']}")
        print(f"   테스트 F1: {result['test_metrics']['f1']:.4f}")

        return 0

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
