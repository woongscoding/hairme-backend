"""
모델 평가 및 자동 배포 스크립트

- 새 모델과 현재 프로덕션 모델 성능 비교
- 성능이 개선되었을 때만 배포
- 배포 히스토리 기록
"""

import os
import sys
import json
import shutil
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# 프로젝트 루트 디렉토리를 Python path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ==================== 모델 정의 (재학습 스크립트와 동일) ====================
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


# ==================== 데이터셋 클래스 ====================
class HairstyleDataset(Dataset):
    """헤어스타일 추천 데이터셋"""

    def __init__(self, csv_path, encoders):
        self.df = pd.read_csv(csv_path)

        self.face_encoded = encoders['face'].transform(self.df['face_shape'])
        self.skin_encoded = encoders['skin'].transform(self.df['skin_tone'])
        self.style_encoded = encoders['style'].transform(self.df['hairstyle'])

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


# ==================== 모델 배포 클래스 ====================
class ModelDeployer:
    """모델 평가 및 배포 클래스"""

    def __init__(
        self,
        new_model_path="models/checkpoints/model_latest.pth",
        new_encoder_path="models/checkpoints/encoders_latest.pkl",
        current_model_path="models/final_model.pth",
        current_encoder_path="models/encoders.pkl",
        test_data_path="data_source/test_data.csv",
        deployment_log_path="models/deployment_history.json"
    ):
        self.project_root = project_root
        self.new_model_path = self.project_root / new_model_path
        self.new_encoder_path = self.project_root / new_encoder_path
        self.current_model_path = self.project_root / current_model_path
        self.current_encoder_path = self.project_root / current_encoder_path
        self.test_data_path = self.project_root / test_data_path
        self.deployment_log_path = self.project_root / deployment_log_path

        self.device = torch.device("cpu")

    def load_encoders(self, encoder_path):
        """인코더 로드"""
        with open(encoder_path, 'rb') as f:
            encoders = pickle.load(f)
        return encoders

    def load_model(self, model_path, encoder_path):
        """모델 로드"""
        # 인코더 로드
        encoders = self.load_encoders(encoder_path)

        # 모델 생성
        n_faces = len(encoders['face'].classes_)
        n_skins = len(encoders['skin'].classes_)
        n_styles = len(encoders['style'].classes_)

        model = HairstyleRecommender(
            n_faces=n_faces,
            n_skins=n_skins,
            n_styles=n_styles
        ).to(self.device)

        # 가중치 로드
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        return model, encoders

    def evaluate_model(self, model, encoders):
        """모델 성능 평가"""
        # 테스트 데이터 로드
        test_dataset = HairstyleDataset(self.test_data_path, encoders)
        test_loader = DataLoader(test_dataset, batch_size=64)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                face = batch['face'].to(self.device)
                skin = batch['skin'].to(self.device)
                style = batch['style'].to(self.device)
                feedback_target = batch['feedback'].to(self.device)

                _, feedback_logits = model(face, skin, style)
                feedback_pred = torch.argmax(feedback_logits, dim=1)

                all_preds.extend(feedback_pred.cpu().numpy())
                all_targets.extend(feedback_target.cpu().numpy())

        # 메트릭 계산
        accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets,
            all_preds,
            average='weighted',
            zero_division=0
        )

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }

        return metrics

    def compare_models(self):
        """새 모델과 현재 모델 비교"""
        print("=" * 60)
        print("📊 모델 성능 비교")
        print("=" * 60)

        # 새 모델 평가
        if not self.new_model_path.exists():
            print("❌ 새 모델 파일이 없습니다.")
            return None

        print("\n🆕 새 모델 평가 중...")
        new_model, new_encoders = self.load_model(self.new_model_path, self.new_encoder_path)
        new_metrics = self.evaluate_model(new_model, new_encoders)

        print(f"   Accuracy: {new_metrics['accuracy']:.4f}")
        print(f"   Precision: {new_metrics['precision']:.4f}")
        print(f"   Recall: {new_metrics['recall']:.4f}")
        print(f"   F1-Score: {new_metrics['f1']:.4f}")

        # 현재 프로덕션 모델 평가 (존재하는 경우)
        current_metrics = None
        if self.current_model_path.exists() and self.current_encoder_path.exists():
            print("\n📦 현재 프로덕션 모델 평가 중...")
            try:
                current_model, current_encoders = self.load_model(
                    self.current_model_path,
                    self.current_encoder_path
                )
                current_metrics = self.evaluate_model(current_model, current_encoders)

                print(f"   Accuracy: {current_metrics['accuracy']:.4f}")
                print(f"   Precision: {current_metrics['precision']:.4f}")
                print(f"   Recall: {current_metrics['recall']:.4f}")
                print(f"   F1-Score: {current_metrics['f1']:.4f}")

            except Exception as e:
                print(f"   ⚠️ 현재 모델 평가 실패: {e}")
                current_metrics = None
        else:
            print("\n⚠️ 현재 프로덕션 모델 없음 (첫 배포)")

        return {
            'new': new_metrics,
            'current': current_metrics
        }

    def should_deploy(self, comparison, min_improvement=0.0):
        """
        배포 여부 결정

        Args:
            comparison: 모델 비교 결과
            min_improvement: 최소 개선 폭 (F1-score 기준)

        Returns:
            bool: 배포 여부
        """
        # 현재 모델이 없으면 무조건 배포
        if comparison['current'] is None:
            print("\n✅ 배포 결정: 첫 배포 (현재 모델 없음)")
            return True

        # F1-score 비교
        new_f1 = comparison['new']['f1']
        current_f1 = comparison['current']['f1']
        improvement = new_f1 - current_f1

        print("\n" + "=" * 60)
        print("🎯 배포 여부 결정")
        print("=" * 60)
        print(f"   현재 F1: {current_f1:.4f}")
        print(f"   새 모델 F1: {new_f1:.4f}")
        print(f"   개선폭: {improvement:+.4f}")
        print(f"   최소 요구 개선폭: {min_improvement:+.4f}")

        if improvement >= min_improvement:
            print(f"\n✅ 배포 결정: 성능 개선됨 ({improvement:+.4f})")
            return True
        else:
            print(f"\n❌ 배포 거부: 성능 개선 부족 ({improvement:+.4f} < {min_improvement:+.4f})")
            return False

    def deploy(self):
        """프로덕션으로 배포"""
        print("\n" + "=" * 60)
        print("🚀 프로덕션 배포 시작")
        print("=" * 60)

        # 기존 모델 백업
        if self.current_model_path.exists():
            backup_dir = self.project_root / "models" / "backups"
            backup_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_model = backup_dir / f"final_model_{timestamp}.pth"
            backup_encoder = backup_dir / f"encoders_{timestamp}.pkl"

            shutil.copy(self.current_model_path, backup_model)
            shutil.copy(self.current_encoder_path, backup_encoder)

            print(f"✅ 기존 모델 백업:")
            print(f"   {backup_model}")
            print(f"   {backup_encoder}")

        # 새 모델을 프로덕션으로 복사
        shutil.copy(self.new_model_path, self.current_model_path)
        shutil.copy(self.new_encoder_path, self.current_encoder_path)

        print(f"\n✅ 프로덕션 배포 완료:")
        print(f"   {self.current_model_path}")
        print(f"   {self.current_encoder_path}")

    def log_deployment(self, comparison, deployed):
        """배포 히스토리 기록"""
        # 기존 로그 로드
        if self.deployment_log_path.exists():
            with open(self.deployment_log_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []

        # 새 배포 기록 추가
        record = {
            'timestamp': datetime.now().isoformat(),
            'deployed': deployed,
            'new_model_metrics': comparison['new'],
            'current_model_metrics': comparison['current']
        }

        history.append(record)

        # 저장
        with open(self.deployment_log_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 배포 히스토리 기록: {self.deployment_log_path}")

    def run(self, min_improvement=0.0, auto_deploy=True):
        """전체 배포 프로세스 실행"""
        print("\n" + "🚀" * 30)
        print("모델 배포 프로세스 시작")
        print("🚀" * 30 + "\n")

        # 1. 모델 비교
        comparison = self.compare_models()

        if comparison is None:
            print("\n❌ 모델 비교 실패")
            return False

        # 2. 배포 여부 결정
        should_deploy = self.should_deploy(comparison, min_improvement)

        # 3. 배포 (자동 또는 수동)
        deployed = False
        if should_deploy:
            if auto_deploy:
                self.deploy()
                deployed = True
            else:
                print("\n⚠️ 자동 배포 비활성화. 수동으로 배포하세요.")
        else:
            print("\n⚠️ 성능 개선이 충분하지 않아 배포하지 않습니다.")

        # 4. 히스토리 기록
        self.log_deployment(comparison, deployed)

        print("\n" + "=" * 60)
        print("✅ 배포 프로세스 완료!")
        print("=" * 60)

        return deployed


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="모델 평가 및 배포")
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.0,
        help="최소 F1-score 개선폭 (기본: 0.0, 조금이라도 개선되면 배포)"
    )
    parser.add_argument(
        "--no-auto-deploy",
        action="store_true",
        help="자동 배포 비활성화 (평가만 수행)"
    )

    args = parser.parse_args()

    try:
        deployer = ModelDeployer()
        deployed = deployer.run(
            min_improvement=args.min_improvement,
            auto_deploy=not args.no_auto_deploy
        )

        if deployed:
            print("\n🎉 새 모델이 프로덕션에 배포되었습니다!")
            print("⚠️ 서버를 재시작하여 새 모델을 로드하세요.")
            return 0
        else:
            print("\n⚠️ 모델이 배포되지 않았습니다.")
            return 1

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
