"""
HairMe ML Trainer Lambda

EventBridge 또는 수동 트리거로 실행됩니다.
S3에서 피드백 데이터를 가져와 모델을 재학습합니다.

학습 파이프라인:
1. S3에서 feedback/pending/*.npz 로드
2. 기존 model.pt 기반 fine-tuning
3. 새 모델 저장:
   - models/current/model.pt (교체)
   - models/archive/v6_feedback_YYYYMMDD.pt (백업)
4. pending/*.npz → processed/로 이동
5. hairme-analyze Lambda 환경변수 업데이트
6. metadata.json 업데이트

Author: HairMe ML Team
Date: 2025-12-02
"""

import json
import os
import io
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
S3_BUCKET = os.getenv('MLOPS_S3_BUCKET', 'hairme-mlops')
MIN_SAMPLES = int(os.getenv('MLOPS_MIN_SAMPLES', '50'))
ANALYZE_LAMBDA_NAME = os.getenv('ANALYZE_LAMBDA_NAME', 'hairme-analyze')
# AWS_REGION은 Lambda 내장 환경변수 사용 (AWS_DEFAULT_REGION)
AWS_REGION = os.getenv('AWS_DEFAULT_REGION', os.getenv('AWS_REGION', 'ap-northeast-2'))

# 학습 하이퍼파라미터
FINE_TUNE_EPOCHS = int(os.getenv('FINE_TUNE_EPOCHS', '10'))
FINE_TUNE_LR = float(os.getenv('FINE_TUNE_LR', '0.0001'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))

# 라벨 정규화 상수
LABEL_MIN = 10.0
LABEL_MAX = 95.0
LABEL_RANGE = LABEL_MAX - LABEL_MIN


# ========== 모델 정의 (RecommendationModelV6 복사) ==========
class MultiTokenAttentionLayer(nn.Module):
    """3-Token Cross-Attention Layer"""

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

        self.face_to_token = nn.Linear(face_dim, token_dim)
        self.skin_to_token = nn.Linear(skin_dim, token_dim)
        self.style_to_token = nn.Linear(style_dim, token_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(token_dim)

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
        face_proj: torch.Tensor,
        skin_proj: torch.Tensor,
        style_emb: torch.Tensor
    ) -> torch.Tensor:
        batch_size = face_proj.size(0)

        face_token = self.face_to_token(face_proj)
        skin_token = self.skin_to_token(skin_proj)
        style_token = self.style_to_token(style_emb)

        tokens = torch.stack([face_token, skin_token, style_token], dim=1)

        attn_out, _ = self.attention(tokens, tokens, tokens)
        tokens = self.norm1(tokens + self.dropout(attn_out))

        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + self.dropout(ffn_out))

        output = tokens.reshape(batch_size, -1)
        return output


class RecommendationModelV6(nn.Module):
    """Multi-Token Attention 기반 추천 모델 v6"""

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

        self.multi_token_attention = MultiTokenAttentionLayer(
            face_dim=64,
            skin_dim=32,
            style_dim=style_embed_dim,
            token_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout_rate * 0.3
        )

        attention_out_dim = token_dim * 3

        self.fc1 = nn.Linear(attention_out_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate * 0.7)

        self.residual_proj = nn.Linear(attention_out_dim, 128)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout_rate * 0.5)

        self.fc4 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        face_features: torch.Tensor,
        skin_features: torch.Tensor,
        style_emb: torch.Tensor
    ) -> torch.Tensor:
        face_proj = self.face_projection(face_features)
        skin_proj = self.skin_projection(skin_features)

        x = self.multi_token_attention(face_proj, skin_proj, style_emb)

        residual = self.residual_proj(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = x + residual

        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = torch.relu(x)

        x = self.fc_out(x)
        x = self.sigmoid(x)

        return x.squeeze(-1)


class FeedbackDataset(Dataset):
    """피드백 데이터셋"""

    def __init__(
        self,
        face_features: np.ndarray,
        skin_features: np.ndarray,
        style_embeddings: np.ndarray,
        ground_truths: np.ndarray
    ):
        self.face_features = torch.tensor(face_features, dtype=torch.float32)
        self.skin_features = torch.tensor(skin_features, dtype=torch.float32)
        self.style_embeddings = torch.tensor(style_embeddings, dtype=torch.float32)

        # 라벨 정규화 (10~95 → 0~1)
        normalized_gt = (ground_truths - LABEL_MIN) / LABEL_RANGE
        self.ground_truths = torch.tensor(normalized_gt, dtype=torch.float32)

    def __len__(self):
        return len(self.ground_truths)

    def __getitem__(self, idx):
        return (
            self.face_features[idx],
            self.skin_features[idx],
            self.style_embeddings[idx],
            self.ground_truths[idx]
        )


def get_s3_client():
    """S3 클라이언트 싱글톤"""
    import boto3
    return boto3.client('s3', region_name=AWS_REGION)


def get_lambda_client():
    """Lambda 클라이언트"""
    import boto3
    return boto3.client('lambda', region_name=AWS_REGION)


def get_pending_count() -> int:
    """S3에서 pending 피드백 수 확인"""
    s3 = get_s3_client()

    try:
        response = s3.get_object(
            Bucket=S3_BUCKET,
            Key='feedback/metadata.json'
        )
        metadata = json.loads(response['Body'].read().decode('utf-8'))
        return metadata.get('pending_count', 0)
    except Exception as e:
        logger.error(f"Failed to get metadata: {e}")
        return 0


def get_metadata() -> Dict[str, Any]:
    """메타데이터 조회"""
    s3 = get_s3_client()

    try:
        response = s3.get_object(
            Bucket=S3_BUCKET,
            Key='feedback/metadata.json'
        )
        return json.loads(response['Body'].read().decode('utf-8'))
    except s3.exceptions.NoSuchKey:
        return {
            "total_feedback_count": 0,
            "pending_count": 0,
            "last_training_at": None,
            "model_version": "v6"
        }
    except Exception as e:
        logger.error(f"Failed to get metadata: {e}")
        return {}


def update_metadata(
    pending_count: int = None,
    training_triggered: bool = False,
    new_model_version: str = None
):
    """메타데이터 업데이트"""
    s3 = get_s3_client()

    try:
        metadata = get_metadata()

        if training_triggered:
            metadata['last_training_at'] = datetime.now(timezone.utc).isoformat()
            metadata['pending_count'] = 0

        if pending_count is not None:
            metadata['pending_count'] = pending_count

        if new_model_version:
            metadata['model_version'] = new_model_version

        s3.put_object(
            Bucket=S3_BUCKET,
            Key='feedback/metadata.json',
            Body=json.dumps(metadata, indent=2, ensure_ascii=False),
            ContentType='application/json'
        )
        logger.info(f"✅ 메타데이터 업데이트 완료: {metadata}")

    except Exception as e:
        logger.error(f"Failed to update metadata: {e}")


def load_pending_feedbacks() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, List[str]]:
    """
    S3에서 pending 피드백 데이터 로드

    Returns:
        (face_features, skin_features, style_embeddings, ground_truths, count, file_keys)
    """
    s3 = get_s3_client()

    try:
        response = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix='feedback/pending/'
        )

        if 'Contents' not in response:
            logger.info("No pending feedbacks found")
            return None, None, None, None, 0, []

        face_list = []
        skin_list = []
        style_list = []
        gt_list = []
        file_keys = []

        for obj in response['Contents']:
            key = obj['Key']
            if not key.endswith('.npz'):
                continue

            try:
                obj_response = s3.get_object(Bucket=S3_BUCKET, Key=key)
                buffer = io.BytesIO(obj_response['Body'].read())
                data = np.load(buffer, allow_pickle=True)

                face_list.append(data['face_features'])
                skin_list.append(data['skin_features'])
                style_list.append(data['style_embedding'])
                gt_list.append(data['ground_truth'])
                file_keys.append(key)

            except Exception as e:
                logger.warning(f"Failed to load {key}: {e}")
                continue

        if not face_list:
            return None, None, None, None, 0, []

        face_features = np.stack(face_list)
        skin_features = np.stack(skin_list)
        style_embeddings = np.stack(style_list)
        ground_truths = np.concatenate(gt_list)

        logger.info(f"✅ {len(face_list)}개 피드백 데이터 로드 완료")

        return face_features, skin_features, style_embeddings, ground_truths, len(face_list), file_keys

    except Exception as e:
        logger.error(f"❌ 피드백 데이터 로드 실패: {e}")
        return None, None, None, None, 0, []


def load_base_model() -> Tuple[Optional[RecommendationModelV6], Dict[str, Any]]:
    """
    S3에서 기존 모델 로드

    Returns:
        (model, config)
    """
    s3 = get_s3_client()

    try:
        # S3에서 현재 모델 다운로드
        response = s3.get_object(
            Bucket=S3_BUCKET,
            Key='models/current/model.pt'
        )

        buffer = io.BytesIO(response['Body'].read())

        # CPU에서 로드
        checkpoint = torch.load(buffer, map_location='cpu', weights_only=False)

        # 설정 추출
        config = checkpoint.get('config', {
            'version': 'v6',
            'token_dim': 128,
            'num_heads': 4,
            'normalized': True
        })

        # 모델 생성 및 가중치 로드
        model = RecommendationModelV6(
            token_dim=config.get('token_dim', 128),
            num_heads=config.get('num_heads', 4)
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"✅ 기존 모델 로드 완료: version={config.get('version', 'unknown')}")
        return model, config

    except s3.exceptions.NoSuchKey:
        logger.warning("⚠️ 기존 모델이 없음 - 새 모델 생성")
        model = RecommendationModelV6()
        config = {
            'version': 'v6',
            'token_dim': 128,
            'num_heads': 4,
            'normalized': True,
            'label_min': LABEL_MIN,
            'label_max': LABEL_MAX,
            'label_range': LABEL_RANGE,
            'attention_type': 'multi_token'
        }
        return model, config

    except Exception as e:
        logger.error(f"❌ 모델 로드 실패: {e}")
        traceback.print_exc()
        return None, {}


def fine_tune_model(
    model: RecommendationModelV6,
    face_features: np.ndarray,
    skin_features: np.ndarray,
    style_embeddings: np.ndarray,
    ground_truths: np.ndarray,
    epochs: int = FINE_TUNE_EPOCHS,
    lr: float = FINE_TUNE_LR
) -> Tuple[RecommendationModelV6, Dict[str, Any]]:
    """
    피드백 데이터로 모델 Fine-tuning

    Returns:
        (fine_tuned_model, training_stats)
    """
    device = torch.device('cpu')  # Lambda는 CPU만 사용
    model = model.to(device)
    model.train()

    # 데이터셋 및 데이터로더
    dataset = FeedbackDataset(face_features, skin_features, style_embeddings, ground_truths)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 옵티마이저 및 손실 함수
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    # 학습 기록
    training_stats = {
        'epochs': epochs,
        'samples': len(dataset),
        'losses': []
    }

    logger.info(f"🏋️ Fine-tuning 시작: {len(dataset)}개 샘플, {epochs} 에폭")

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for face, skin, style, gt in dataloader:
            face = face.to(device)
            skin = skin.to(device)
            style = style.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()
            pred = model(face, skin, style)
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        training_stats['losses'].append(avg_loss)

        if (epoch + 1) % 2 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch + 1}/{epochs}: loss = {avg_loss:.4f}")

    model.eval()
    training_stats['final_loss'] = training_stats['losses'][-1] if training_stats['losses'] else 0

    logger.info(f"✅ Fine-tuning 완료: final_loss = {training_stats['final_loss']:.4f}")

    return model, training_stats


def save_model_to_s3(
    model: RecommendationModelV6,
    config: Dict[str, Any],
    new_version: str
) -> bool:
    """
    학습된 모델을 S3에 저장

    - models/current/model.pt: 현재 모델 (교체)
    - models/archive/{new_version}.pt: 아카이브

    Returns:
        성공 여부
    """
    s3 = get_s3_client()

    try:
        # 체크포인트 생성
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'version': new_version,
            'trained_at': datetime.now(timezone.utc).isoformat()
        }

        # 바이너리로 직렬화
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        model_bytes = buffer.getvalue()

        # 1. 아카이브에 백업
        archive_key = f'models/archive/{new_version}.pt'
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=archive_key,
            Body=model_bytes,
            ContentType='application/octet-stream'
        )
        logger.info(f"✅ 모델 아카이브 저장: {archive_key}")

        # 2. 현재 모델 교체
        s3.put_object(
            Bucket=S3_BUCKET,
            Key='models/current/model.pt',
            Body=model_bytes,
            ContentType='application/octet-stream'
        )
        logger.info(f"✅ 현재 모델 교체 완료")

        return True

    except Exception as e:
        logger.error(f"❌ 모델 저장 실패: {e}")
        traceback.print_exc()
        return False


def move_pending_to_processed(file_keys: List[str], batch_name: str) -> bool:
    """
    pending 파일들을 processed로 이동

    Returns:
        성공 여부
    """
    s3 = get_s3_client()

    try:
        moved_count = 0
        for old_key in file_keys:
            filename = old_key.split('/')[-1]
            new_key = f'feedback/processed/{batch_name}/{filename}'

            # Copy then delete
            s3.copy_object(
                Bucket=S3_BUCKET,
                CopySource={'Bucket': S3_BUCKET, 'Key': old_key},
                Key=new_key
            )
            s3.delete_object(Bucket=S3_BUCKET, Key=old_key)
            moved_count += 1

        logger.info(f"✅ {moved_count}개 파일을 processed로 이동: {batch_name}")
        return True

    except Exception as e:
        logger.error(f"❌ 파일 이동 실패: {e}")
        return False


def backup_lambda_config() -> Optional[Dict[str, Any]]:
    """
    hairme-analyze Lambda의 현재 환경변수 백업

    Returns:
        현재 환경변수 또는 None
    """
    lambda_client = get_lambda_client()

    try:
        response = lambda_client.get_function_configuration(
            FunctionName=ANALYZE_LAMBDA_NAME
        )
        env_vars = response.get('Environment', {}).get('Variables', {})

        # S3에 백업
        s3 = get_s3_client()
        backup_key = f'config_backups/{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.json'
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=backup_key,
            Body=json.dumps(env_vars, indent=2),
            ContentType='application/json'
        )
        logger.info(f"✅ Lambda 환경변수 백업: {backup_key}")

        return env_vars

    except Exception as e:
        logger.error(f"❌ Lambda 설정 백업 실패: {e}")
        return None


def update_analyze_lambda_envvars(new_version: str, experiment_id: str) -> bool:
    """
    hairme-analyze Lambda 환경변수 업데이트

    - ABTEST_CHALLENGER_VERSION: 새 모델 버전
    - ABTEST_EXPERIMENT_ID: 새 실험 ID

    Returns:
        성공 여부
    """
    lambda_client = get_lambda_client()

    try:
        # 현재 설정 조회
        response = lambda_client.get_function_configuration(
            FunctionName=ANALYZE_LAMBDA_NAME
        )
        current_env = response.get('Environment', {}).get('Variables', {})

        # 환경변수 업데이트
        current_env['ABTEST_CHALLENGER_VERSION'] = new_version
        current_env['ABTEST_EXPERIMENT_ID'] = experiment_id

        # Lambda 업데이트
        lambda_client.update_function_configuration(
            FunctionName=ANALYZE_LAMBDA_NAME,
            Environment={'Variables': current_env}
        )

        logger.info(
            f"✅ Lambda 환경변수 업데이트 완료: "
            f"CHALLENGER={new_version}, EXPERIMENT_ID={experiment_id}"
        )
        return True

    except Exception as e:
        logger.error(f"❌ Lambda 환경변수 업데이트 실패: {e}")
        traceback.print_exc()
        return False


def run_training_pipeline() -> Dict[str, Any]:
    """
    전체 학습 파이프라인 실행

    Returns:
        결과 딕셔너리
    """
    timestamp = datetime.now(timezone.utc)
    date_str = timestamp.strftime('%Y%m%d')
    new_version = f'v6_feedback_{date_str}'
    experiment_id = f'exp_{timestamp.strftime("%Y_%m_%d")}'
    batch_name = f'batch_{timestamp.strftime("%Y%m%d_%H%M%S")}'

    result = {
        'success': False,
        'new_version': new_version,
        'experiment_id': experiment_id,
        'samples_trained': 0,
        'final_loss': None,
        'steps_completed': []
    }

    try:
        # 1. 피드백 데이터 로드
        logger.info("📥 Step 1: 피드백 데이터 로드")
        face, skin, style, gt, count, file_keys = load_pending_feedbacks()

        if count == 0:
            result['message'] = 'No pending feedbacks'
            return result

        result['samples_trained'] = count
        result['steps_completed'].append('load_feedbacks')

        # 2. 기존 모델 로드
        logger.info("📥 Step 2: 기존 모델 로드")
        model, config = load_base_model()

        if model is None:
            result['message'] = 'Failed to load base model'
            return result

        result['steps_completed'].append('load_model')

        # 3. Fine-tuning
        logger.info("🏋️ Step 3: Fine-tuning")
        model, stats = fine_tune_model(model, face, skin, style, gt)
        result['final_loss'] = stats['final_loss']
        result['steps_completed'].append('fine_tune')

        # 4. 설정 업데이트
        config['version'] = new_version
        config['fine_tuned_at'] = timestamp.isoformat()
        config['samples_count'] = count

        # 5. Lambda 환경변수 백업
        logger.info("💾 Step 4: Lambda 설정 백업")
        backup_lambda_config()
        result['steps_completed'].append('backup_config')

        # 6. 모델 저장
        logger.info("💾 Step 5: 모델 저장")
        if not save_model_to_s3(model, config, new_version):
            result['message'] = 'Failed to save model'
            return result

        result['steps_completed'].append('save_model')

        # 7. pending → processed 이동
        logger.info("📦 Step 6: 피드백 파일 이동")
        move_pending_to_processed(file_keys, batch_name)
        result['steps_completed'].append('move_feedbacks')

        # 8. Lambda 환경변수 업데이트
        logger.info("🔧 Step 7: Lambda 환경변수 업데이트")
        if not update_analyze_lambda_envvars(new_version, experiment_id):
            result['message'] = 'Model saved but Lambda update failed'
            # 모델은 저장되었으므로 부분 성공으로 처리
            result['success'] = True
            result['steps_completed'].append('lambda_update_failed')
            return result

        result['steps_completed'].append('update_lambda')

        # 9. 메타데이터 업데이트
        logger.info("📝 Step 8: 메타데이터 업데이트")
        update_metadata(training_triggered=True, new_model_version=new_version)
        result['steps_completed'].append('update_metadata')

        result['success'] = True
        result['message'] = 'Training completed successfully'

        logger.info(f"✅ 학습 파이프라인 완료: {new_version}")

        return result

    except Exception as e:
        logger.error(f"❌ 학습 파이프라인 실패: {e}")
        traceback.print_exc()
        result['message'] = str(e)
        return result


def lambda_handler(event, context):
    """
    Lambda 핸들러

    Args:
        event: {
            "trigger_type": "scheduled" | "data_threshold" | "manual",
            "force": false,  # true이면 MIN_SAMPLES 무시
            "metadata": {...}
        }

    Returns:
        {
            'statusCode': 200 | 500,
            'body': {
                'success': bool,
                'message': str,
                'trigger_type': str,
                'pending_count': int,
                'training_result': {...}  # 학습 수행 시
            }
        }
    """
    logger.info(f"🚀 Trainer Lambda 시작")
    logger.info(f"Event: {json.dumps(event)}")

    trigger_type = event.get('trigger_type', 'unknown')
    force_train = event.get('force', False)
    timestamp = datetime.now(timezone.utc).isoformat()

    # Pending 피드백 수 확인
    pending_count = get_pending_count()
    logger.info(f"📊 Pending feedback count: {pending_count}")

    # 최소 샘플 수 확인 (force가 아닐 때)
    if not force_train and pending_count < MIN_SAMPLES:
        message = f"Insufficient data: {pending_count}/{MIN_SAMPLES} samples"
        logger.info(f"⏸️ {message}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': False,
                'message': message,
                'trigger_type': trigger_type,
                'pending_count': pending_count,
                'min_samples': MIN_SAMPLES,
                'timestamp': timestamp
            })
        }

    # 실제 학습 파이프라인 실행
    logger.info(f"🏋️ Training triggered with {pending_count} samples (force={force_train})")

    try:
        training_result = run_training_pipeline()

        if training_result['success']:
            logger.info(f"✅ 학습 완료: {training_result['new_version']}")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'success': True,
                    'message': 'Training completed successfully',
                    'trigger_type': trigger_type,
                    'pending_count': pending_count,
                    'training_result': training_result,
                    'timestamp': timestamp
                })
            }
        else:
            logger.error(f"❌ 학습 실패: {training_result.get('message')}")
            return {
                'statusCode': 200,  # Lambda 자체는 성공, 학습만 실패
                'body': json.dumps({
                    'success': False,
                    'message': training_result.get('message', 'Training failed'),
                    'trigger_type': trigger_type,
                    'pending_count': pending_count,
                    'training_result': training_result,
                    'timestamp': timestamp
                })
            }

    except Exception as e:
        logger.error(f"❌ Lambda 실행 중 예외 발생: {e}")
        traceback.print_exc()

        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'message': str(e),
                'trigger_type': trigger_type,
                'pending_count': pending_count,
                'timestamp': timestamp
            })
        }
