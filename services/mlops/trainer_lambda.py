"""
Trainer Lambda 함수

S3에서 피드백 데이터를 가져와 모델을 재학습합니다.

실행 환경:
    - Lambda (15분 제한, 10GB 메모리) 또는
    - EC2 Spot Instance (대용량 학습)

Flow:
    1. S3에서 pending 피드백 데이터 로드
    2. 기존 학습 데이터와 병합
    3. 모델 재학습 (incremental 또는 full)
    4. 새 모델을 S3에 업로드
    5. Lambda 함수 업데이트 트리거 (선택)

비용:
    - Lambda (10GB, 15분): ~$0.15/실행
    - EC2 Spot t3.medium: ~$0.01/시간

Author: HairMe ML Team
Date: 2025-12-02
"""

import os
import io
import json
import tempfile
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Configuration
S3_BUCKET = os.getenv("MLOPS_S3_BUCKET", "hairme-mlops")
MODEL_VERSION_PREFIX = "v6"
MIN_SAMPLES_FOR_TRAINING = 50  # 최소 학습 샘플 수

try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelTrainer:
    """
    ML 모델 트레이너

    Lambda 또는 EC2에서 실행되어 피드백 기반 재학습을 수행합니다.
    """

    def __init__(self, s3_bucket: str = S3_BUCKET):
        """초기화"""
        self.s3_bucket = s3_bucket
        self.s3_client = None
        self.device = None

        if BOTO3_AVAILABLE:
            self.s3_client = boto3.client("s3")

        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"🔧 PyTorch device: {self.device}")

    def load_pending_data(self) -> Tuple[Optional[np.ndarray], ...]:
        """
        S3에서 pending 피드백 데이터 로드

        Returns:
            (face_features, skin_features, style_embeddings, ground_truths)
        """
        if not self.s3_client:
            return None, None, None, None

        try:
            # pending 폴더의 모든 NPZ 파일 목록
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket, Prefix="feedback/pending/"
            )

            if "Contents" not in response:
                logger.info("ℹ️ 대기 중인 피드백 없음")
                return None, None, None, None

            face_list = []
            skin_list = []
            style_list = []
            gt_list = []

            for obj in response["Contents"]:
                key = obj["Key"]
                if not key.endswith(".npz"):
                    continue

                # NPZ 파일 다운로드
                obj_response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=key)

                buffer = io.BytesIO(obj_response["Body"].read())
                data = np.load(buffer, allow_pickle=True)

                face_list.append(data["face_features"])
                skin_list.append(data["skin_features"])
                style_list.append(data["style_embedding"])
                gt_list.append(data["ground_truth"])

            if not face_list:
                return None, None, None, None

            logger.info(f"✅ {len(face_list)}개 피드백 데이터 로드")

            return (
                np.stack(face_list),
                np.stack(skin_list),
                np.stack(style_list),
                np.concatenate(gt_list),
            )

        except Exception as e:
            logger.error(f"❌ 데이터 로드 실패: {e}")
            return None, None, None, None

    def load_base_model(self) -> Optional[nn.Module]:
        """
        S3에서 현재 운영 모델 로드

        Returns:
            PyTorch 모델 또는 None
        """
        if not TORCH_AVAILABLE or not self.s3_client:
            return None

        try:
            # 현재 모델 다운로드
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                self.s3_client.download_file(
                    self.s3_bucket, "models/current/model.pt", tmp.name
                )

                checkpoint = torch.load(
                    tmp.name, map_location=self.device, weights_only=False
                )

                # 모델 클래스 동적 임포트
                from models.ml_recommender import RecommendationModelV6

                model = RecommendationModelV6()

                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)

                model.to(self.device)
                logger.info("✅ 기존 모델 로드 완료")

                # 임시 파일 삭제
                os.unlink(tmp.name)

                return model

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.warning("⚠️ 기존 모델 없음 - 새로 학습 시작")
                return None
            raise
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            return None

    def train(
        self,
        face_features: np.ndarray,
        skin_features: np.ndarray,
        style_embeddings: np.ndarray,
        ground_truths: np.ndarray,
        base_model: Optional[nn.Module] = None,
        epochs: int = 10,
        learning_rate: float = 0.0001,
        batch_size: int = 32,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        모델 학습

        Args:
            face_features: (N, 6) 얼굴 특징
            skin_features: (N, 2) 피부 특징
            style_embeddings: (N, 384) 스타일 임베딩
            ground_truths: (N,) 정답 레이블
            base_model: 기존 모델 (없으면 새로 생성)
            epochs: 학습 에폭
            learning_rate: 학습률
            batch_size: 배치 크기

        Returns:
            (trained_model, training_history)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")

        # 라벨 정규화 (10~95 -> 0~1)
        LABEL_MIN, LABEL_MAX = 10.0, 95.0
        normalized_labels = (ground_truths - LABEL_MIN) / (LABEL_MAX - LABEL_MIN)

        # 텐서 변환
        face_tensor = torch.FloatTensor(face_features).to(self.device)
        skin_tensor = torch.FloatTensor(skin_features).to(self.device)
        style_tensor = torch.FloatTensor(style_embeddings).to(self.device)
        label_tensor = torch.FloatTensor(normalized_labels).to(self.device)

        # DataLoader 생성
        dataset = TensorDataset(face_tensor, skin_tensor, style_tensor, label_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 모델 생성 또는 로드
        if base_model is None:
            from models.ml_recommender import RecommendationModelV6

            model = RecommendationModelV6()
            model.to(self.device)
            logger.info("🆕 새 모델 생성")
        else:
            model = base_model
            logger.info("📂 기존 모델에서 fine-tuning")

        model.train()

        # 손실 함수 및 옵티마이저
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 학습 히스토리
        history = {
            "epochs": epochs,
            "losses": [],
            "samples": len(ground_truths),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        # 학습 루프
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            for face_batch, skin_batch, style_batch, label_batch in dataloader:
                optimizer.zero_grad()

                predictions = model(face_batch, skin_batch, style_batch)
                loss = criterion(predictions, label_batch)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / batch_count
            history["losses"].append(avg_loss)

            if (epoch + 1) % 2 == 0:
                logger.info(f"  Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

        history["final_loss"] = history["losses"][-1]
        history["finished_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"✅ 학습 완료: final_loss={history['final_loss']:.6f}")

        return model, history

    def save_model(
        self, model: nn.Module, history: Dict[str, Any], version: Optional[str] = None
    ) -> str:
        """
        학습된 모델을 S3에 저장

        Args:
            model: 학습된 모델
            history: 학습 히스토리
            version: 모델 버전 (없으면 자동 생성)

        Returns:
            S3 키
        """
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")

        # 버전 생성
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version = version or f"{MODEL_VERSION_PREFIX}_{timestamp}"

        # 체크포인트 생성
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": {
                "version": "v6",
                "normalized": True,
                "attention_type": "multi_token",
                "token_dim": 128,
                "num_heads": 4,
            },
            "training_history": history,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            torch.save(checkpoint, tmp.name)

            # 현재 모델로 업로드
            current_key = "models/current/model.pt"
            self.s3_client.upload_file(tmp.name, self.s3_bucket, current_key)

            # 아카이브에도 저장
            archive_key = f"models/archive/{version}.pt"
            self.s3_client.upload_file(tmp.name, self.s3_bucket, archive_key)

            # 임시 파일 삭제
            os.unlink(tmp.name)

        # 메타데이터 업데이트
        metadata = {
            "version": version,
            "samples_trained": history.get("samples", 0),
            "final_loss": history.get("final_loss", 0),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key="models/current/metadata.json",
            Body=json.dumps(metadata, indent=2),
            ContentType="application/json",
        )

        logger.info(f"✅ 모델 저장 완료: {current_key}")

        return current_key

    def mark_data_processed(self):
        """pending 데이터를 processed로 이동"""
        if not self.s3_client:
            return

        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
            batch_name = f"batch_{timestamp}"

            # pending 파일 목록
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket, Prefix="feedback/pending/"
            )

            if "Contents" not in response:
                return

            for obj in response["Contents"]:
                old_key = obj["Key"]
                if not old_key.endswith(".npz"):
                    continue

                filename = old_key.split("/")[-1]
                new_key = f"feedback/processed/{batch_name}/{filename}"

                # Copy then delete
                self.s3_client.copy_object(
                    Bucket=self.s3_bucket,
                    CopySource={"Bucket": self.s3_bucket, "Key": old_key},
                    Key=new_key,
                )
                self.s3_client.delete_object(Bucket=self.s3_bucket, Key=old_key)

            # 메타데이터 업데이트
            try:
                metadata_response = self.s3_client.get_object(
                    Bucket=self.s3_bucket, Key="feedback/metadata.json"
                )
                metadata = json.loads(metadata_response["Body"].read().decode("utf-8"))
            except Exception:
                metadata = {}

            metadata["pending_count"] = 0
            metadata["last_training_at"] = datetime.now(timezone.utc).isoformat()

            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key="feedback/metadata.json",
                Body=json.dumps(metadata, indent=2),
                ContentType="application/json",
            )

            logger.info(f"✅ 데이터 processed로 이동: {batch_name}")

        except Exception as e:
            logger.error(f"❌ 데이터 이동 실패: {e}")


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda 핸들러

    EventBridge 또는 직접 호출로 트리거됩니다.

    Args:
        event: {
            "trigger_type": "scheduled" | "data_threshold" | "manual",
            "metadata": {...}
        }
        context: Lambda context

    Returns:
        {
            "success": bool,
            "message": str,
            "model_version": str (if success)
        }
    """
    logger.info(f"🚀 Trainer Lambda 시작: {json.dumps(event)}")

    trigger_type = event.get("trigger_type", "unknown")

    try:
        trainer = ModelTrainer()

        # 1. 데이터 로드
        face, skin, style, labels = trainer.load_pending_data()

        if face is None or len(face) < MIN_SAMPLES_FOR_TRAINING:
            sample_count = len(face) if face is not None else 0
            logger.info(
                f"ℹ️ 학습 데이터 부족: {sample_count}/{MIN_SAMPLES_FOR_TRAINING}"
            )
            return {
                "success": False,
                "message": f"Insufficient data ({sample_count}/{MIN_SAMPLES_FOR_TRAINING})",
                "trigger_type": trigger_type,
            }

        logger.info(f"📊 학습 데이터: {len(labels)}개 샘플")

        # 2. 기존 모델 로드
        base_model = trainer.load_base_model()

        # 3. 학습
        model, history = trainer.train(
            face_features=face,
            skin_features=skin,
            style_embeddings=style,
            ground_truths=labels,
            base_model=base_model,
            epochs=10,
            learning_rate=0.0001,
        )

        # 4. 모델 저장
        model_key = trainer.save_model(model, history)

        # 5. 데이터 이동
        trainer.mark_data_processed()

        logger.info(f"✅ 재학습 완료: {model_key}")

        return {
            "success": True,
            "message": "Training completed successfully",
            "trigger_type": trigger_type,
            "model_key": model_key,
            "samples_trained": len(labels),
            "final_loss": history.get("final_loss", 0),
        }

    except Exception as e:
        logger.error(f"❌ 재학습 실패: {e}")
        import traceback

        traceback.print_exc()

        return {
            "success": False,
            "message": f"Training failed: {str(e)}",
            "trigger_type": trigger_type,
        }


# 로컬 테스트용
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    result = lambda_handler(event={"trigger_type": "manual"}, context=None)

    print(json.dumps(result, indent=2))
