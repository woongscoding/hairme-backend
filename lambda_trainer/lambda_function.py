"""
HairMe ML Trainer Lambda

EventBridge 또는 수동 트리거로 실행됩니다.
S3에서 피드백 데이터를 가져와 모델을 재학습합니다.

학습 파이프라인:
1. S3에서 feedback/pending/*.npz 로드
2. analysis_id 해시로 학습/홀드아웃 결정적 분할 (홀드아웃은 학습에 쓰지 않음)
3. 기존 model.pt 기반 fine-tuning (학습 분할만 사용)
4. 품질 게이트: 같은 홀드아웃으로 학습 전/후 모델 비교
   - MSE(정규화 공간) 와 pairwise ranking accuracy 가 모두 허용치 안이면 통과
5. 통과 시에만 새 모델 저장:
   - models/current/model.pt (교체)
   - models/archive/v6_feedback_YYYYMMDD.pt (백업)
   거부 시 models/rejected/{version}.pt 에만 저장하고 현재 모델은 유지
6. 통과 시에만 pending/*.npz → processed/로 이동
   (거부 시 pending 은 그대로 두어 다음 학습에 재포함)
7. metadata.json 업데이트

이벤트 플래그:
- force: MIN_SAMPLES 게이트 우회 (품질 게이트는 우회하지 않음)
- skip_gate: 품질 게이트 우회 (from_base 전체 재학습처럼 의도적일 때만)
- allow_random_init: 시작점 모델이 없을 때 랜덤 초기화 허용
- from_base: models/base/model.pt(번들 v6)에서 재학습 (fine-tune 드리프트 누적 차단)
- include_processed: feedback/processed/ 데이터도 학습에 포함 (전체 재학습)

Fine-tuning 중 BatchNorm running stats 는 항상 고정된다(set_batchnorm_eval).
소량 피드백으로 running_var 가 0 에 수렴해 서빙 정규화가 폭주하는 것을 막는다.

Author: HairMe ML Team
Date: 2025-12-02
"""

import json
import os
import io
import hashlib
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
S3_BUCKET = os.getenv("MLOPS_S3_BUCKET", "hairme-mlops")
MIN_SAMPLES = int(os.getenv("MLOPS_MIN_SAMPLES", "50"))
# AWS_REGION은 Lambda 내장 환경변수 사용 (AWS_DEFAULT_REGION)
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "ap-northeast-2"))

# 학습 하이퍼파라미터
FINE_TUNE_EPOCHS = int(os.getenv("FINE_TUNE_EPOCHS", "10"))
FINE_TUNE_LR = float(os.getenv("FINE_TUNE_LR", "0.0001"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

# 품질 게이트 (홀드아웃 평가)
# 홀드아웃 비율: analysis_id 해시로 분할하므로 실제 비율은 목표치 근사값이다.
TRAIN_HOLDOUT_RATIO = float(os.getenv("TRAIN_HOLDOUT_RATIO", "0.15"))
# MSE 는 이전 대비 이 비율만큼 나빠지는 것까지 허용 (0.02 = 2%)
GATE_MSE_TOLERANCE = float(os.getenv("GATE_MSE_TOLERANCE", "0.02"))
# ranking accuracy 는 이전 대비 이 절대값만큼 떨어지는 것까지 허용
GATE_RANK_TOLERANCE = float(os.getenv("GATE_RANK_TOLERANCE", "0.02"))
# 홀드아웃이 이보다 작으면 평가를 신뢰할 수 없으므로 교체를 거부한다
MIN_HOLDOUT_SAMPLES = int(os.getenv("MIN_HOLDOUT_SAMPLES", "20"))

# S3 키 상수
CURRENT_MODEL_KEY = "models/current/model.pt"
BASE_MODEL_KEY = "models/base/model.pt"
REJECTED_MODEL_PREFIX = "models/rejected/"
PENDING_PREFIX = "feedback/pending/"
PROCESSED_PREFIX = "feedback/processed/"

# 라벨 정규화 상수
LABEL_MIN = 10.0
LABEL_MAX = 95.0
LABEL_RANGE = LABEL_MAX - LABEL_MIN

# 체크포인트 config 기본값 (번들 v6 체크포인트와 동일한 키 구성)
DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "version": "v6",
    "face_feat_dim": 6,
    "skin_feat_dim": 2,
    "style_embed_dim": 384,
    "token_dim": 128,
    "num_heads": 4,
    "normalized": True,
    "label_min": LABEL_MIN,
    "label_max": LABEL_MAX,
    "label_range": LABEL_RANGE,
    "attention_type": "multi_token",
}


# ========== 학습 데이터 특징 통계 (입력 스케일링용) ==========
# models/ml_recommender.py 의 FACE_FEATURE_STATS / SKIN_FEATURE_STATS 와
# 반드시 수치적으로 동일하게 유지해야 한다 (train/serve skew 방지).
# ai_face_1000.npz에서 추출한 통계 (5910 샘플)
FACE_FEATURE_STATS = {
    0: {"min": 0.99, "max": 1.51, "mean": 1.20, "std": 0.06},  # face_ratio
    1: {
        "min": 301.10,
        "max": 495.30,
        "mean": 458.13,
        "std": 14.31,
    },  # forehead_width (pixel)
    2: {
        "min": 421.40,
        "max": 641.00,
        "mean": 561.34,
        "std": 19.73,
    },  # cheekbone_width (pixel)
    3: {
        "min": 333.90,
        "max": 524.10,
        "mean": 447.70,
        "std": 19.82,
    },  # jaw_width (pixel)
    4: {"min": 0.71, "max": 0.89, "mean": 0.82, "std": 0.02},  # forehead_ratio
    5: {"min": 0.73, "max": 0.86, "mean": 0.80, "std": 0.02},  # jaw_ratio
}

SKIN_FEATURE_STATS = {
    0: {"min": 50.53, "max": 89.26, "mean": 79.91, "std": 3.90},  # ITA_value
    1: {"min": 5.96, "max": 142.39, "mean": 12.09, "std": 10.97},  # hue_value
}


def scale_input_features(
    face_features: np.ndarray, skin_features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    입력 특징을 학습 데이터 분포에 맞게 스케일링

    serving(models/ml_recommender.scale_input_features)과 동일한 연산이어야 한다.
    - 픽셀 기반 특징(1: forehead_width, 2: cheekbone_width, 3: jaw_width)을
      cheekbone_width 기준으로 학습 데이터 평균 스케일(561.34)로 변환
    - 비율 기반 특징(0, 4, 5)과 피부 특징은 스케일 불변 -> 클리핑만 수행
    - cheekbone_width <= 0 이면 scale_factor = 1.0 (안전값)
    - 이미 스케일링된 값에 다시 적용해도 결과가 동일하다 (idempotent)

    Args:
        face_features: [face_ratio, forehead_width, cheekbone_width,
                        jaw_width, forehead_ratio, jaw_ratio]
        skin_features: [ITA_value, hue_value]

    Returns:
        (scaled_face_features, scaled_skin_features)
    """
    face_scaled = face_features.copy()
    skin_scaled = skin_features.copy()

    input_cheekbone = face_features[2]
    train_cheekbone_mean = FACE_FEATURE_STATS[2]["mean"]  # 561.34

    if input_cheekbone > 0:
        scale_factor = train_cheekbone_mean / input_cheekbone
    else:
        scale_factor = 1.0

    # 픽셀 기반 특징만 스케일링 (인덱스 1, 2, 3)
    face_scaled[1] = face_features[1] * scale_factor  # forehead_width
    face_scaled[2] = face_features[2] * scale_factor  # cheekbone_width
    face_scaled[3] = face_features[3] * scale_factor  # jaw_width

    # 스케일링된 값이 학습 데이터 범위 내에 있도록 클리핑
    for idx in [1, 2, 3]:
        min_val = FACE_FEATURE_STATS[idx]["min"]
        max_val = FACE_FEATURE_STATS[idx]["max"]
        face_scaled[idx] = np.clip(face_scaled[idx], min_val, max_val)

    # 비율 특징도 학습 데이터 범위 내에 있도록 클리핑
    for idx in [0, 4, 5]:
        min_val = FACE_FEATURE_STATS[idx]["min"]
        max_val = FACE_FEATURE_STATS[idx]["max"]
        face_scaled[idx] = np.clip(face_scaled[idx], min_val, max_val)

    # 피부 특징 클리핑 (이미 스케일 불변)
    for idx in [0, 1]:
        min_val = SKIN_FEATURE_STATS[idx]["min"]
        max_val = SKIN_FEATURE_STATS[idx]["max"]
        skin_scaled[idx] = np.clip(skin_scaled[idx], min_val, max_val)

    return face_scaled, skin_scaled


def scale_feature_batch(
    face_features: np.ndarray, skin_features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    (N, 6) / (N, 2) 배치에 scale_input_features를 샘플 단위로 적용

    피드백 NPZ에는 DynamoDB 원본 픽셀값이 그대로 저장되므로,
    학습/평가 직전에 serving과 동일한 스케일로 맞춘다.
    레거시 라벨 인코딩(one-hot) 샘플처럼 차원이 다르면 원본을 그대로 반환한다.
    """
    face_arr = np.asarray(face_features, dtype=np.float32)
    skin_arr = np.asarray(skin_features, dtype=np.float32)

    if face_arr.ndim != 2 or face_arr.shape[1] != 6:
        logger.warning(
            f"⚠️ 예상치 못한 face_features shape={face_arr.shape} - 스케일링 건너뜀"
        )
        return face_arr, skin_arr
    if skin_arr.ndim != 2 or skin_arr.shape[1] != 2:
        logger.warning(
            f"⚠️ 예상치 못한 skin_features shape={skin_arr.shape} - 스케일링 건너뜀"
        )
        return face_arr, skin_arr

    face_out = np.empty_like(face_arr)
    skin_out = np.empty_like(skin_arr)

    for i in range(face_arr.shape[0]):
        face_out[i], skin_out[i] = scale_input_features(face_arr[i], skin_arr[i])

    return face_out, skin_out


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
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_dim = token_dim

        self.face_to_token = nn.Linear(face_dim, token_dim)
        self.skin_to_token = nn.Linear(skin_dim, token_dim)
        self.style_to_token = nn.Linear(style_dim, token_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(token_dim)

        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.norm2 = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, face_proj: torch.Tensor, skin_proj: torch.Tensor, style_emb: torch.Tensor
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
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.face_feat_dim = face_feat_dim
        self.skin_feat_dim = skin_feat_dim
        self.style_embed_dim = style_embed_dim

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

        self.multi_token_attention = MultiTokenAttentionLayer(
            face_dim=64,
            skin_dim=32,
            style_dim=style_embed_dim,
            token_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout_rate * 0.3,
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
        style_emb: torch.Tensor,
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
        ground_truths: np.ndarray,
    ):
        # 입력 스케일링 (serving과 동일한 분포로 정렬 - train/serve skew 방지)
        # NPZ에는 DynamoDB 원본 픽셀값이 저장되므로 텐서 변환 전에 반드시 적용한다.
        scaled_face, scaled_skin = scale_feature_batch(face_features, skin_features)

        self.face_features = torch.tensor(scaled_face, dtype=torch.float32)
        self.skin_features = torch.tensor(scaled_skin, dtype=torch.float32)
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
            self.ground_truths[idx],
        )


# ========== 홀드아웃 분할 (analysis_id 해시 기준) ==========


def analysis_id_from_key(key: str) -> str:
    """
    피드백 NPZ 키에서 analysis_id 그룹 키를 추출한다 (metadata 를 못 읽을 때의 대체 경로).

    파일명 규칙(services/mlops/s3_feedback_store.py):
        {YYYY-MM-DD}_{analysis_id[:8]}_s{style}_{uuid6}.npz
        {YYYY-MM-DD}_trending_{analysis_id[:8]}_s{style}_{uuid6}.npz

    규칙에 맞지 않으면 파일명 전체를 그룹 키로 쓴다(= 자기 자신만의 그룹).
    """
    filename = key.split("/")[-1]
    if filename.endswith(".npz"):
        filename = filename[: -len(".npz")]

    parts = filename.split("_")
    if len(parts) >= 3 and parts[1] == "trending":
        return parts[2]
    if len(parts) >= 2:
        return parts[1]
    return filename


def analysis_id_bucket(analysis_id: str) -> float:
    """
    analysis_id 를 [0, 1) 구간의 결정적 실수로 매핑한다.

    SHA-256 상위 64비트만 쓰므로 파이썬 hash() 의 프로세스별 랜덤 시드와 달리
    실행·리전·샘플 순서와 무관하게 항상 같은 값이 나온다.
    """
    digest = hashlib.sha256(analysis_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def split_holdout_indices(
    analysis_ids: List[str],
    ratio: float = TRAIN_HOLDOUT_RATIO,
) -> Tuple[List[int], List[int]]:
    """
    analysis_id 해시로 학습/홀드아웃 인덱스를 결정적으로 분할한다.

    같은 analysis_id 의 샘플은 항상 같은 쪽으로 간다. pairwise ranking 지표가
    "같은 analysis 안에서" 정의되므로 그룹이 쪼개지면 쌍이 사라지고,
    같은 얼굴의 다른 스타일이 학습/평가에 동시에 들어가 누수가 생긴다.

    Args:
        analysis_ids: 샘플별 analysis_id (길이 = 샘플 수)
        ratio: 홀드아웃 목표 비율. 0 이하이면 전부 학습에 사용한다.

    Returns:
        (train_indices, holdout_indices) - 둘 다 오름차순
    """
    if ratio <= 0:
        return list(range(len(analysis_ids))), []

    train_indices: List[int] = []
    holdout_indices: List[int] = []

    for idx, analysis_id in enumerate(analysis_ids):
        if analysis_id_bucket(analysis_id) < ratio:
            holdout_indices.append(idx)
        else:
            train_indices.append(idx)

    return train_indices, holdout_indices


def take_indices(
    arrays: Tuple[np.ndarray, ...], indices: List[int]
) -> Tuple[np.ndarray, ...]:
    """numpy 배열 튜플에서 같은 인덱스 집합을 잘라낸다"""
    index_array = np.asarray(indices, dtype=np.int64)
    return tuple(arr[index_array] for arr in arrays)


def get_s3_client():
    """S3 클라이언트 싱글톤"""
    import boto3

    return boto3.client("s3", region_name=AWS_REGION)


def get_pending_count() -> int:
    """S3에서 pending 피드백 수 확인"""
    s3 = get_s3_client()

    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key="feedback/metadata.json")
        metadata = json.loads(response["Body"].read().decode("utf-8"))
        return metadata.get("pending_count", 0)
    except Exception as e:
        logger.error(f"Failed to get metadata: {e}")
        return 0


def get_metadata() -> Dict[str, Any]:
    """메타데이터 조회"""
    s3 = get_s3_client()

    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key="feedback/metadata.json")
        return json.loads(response["Body"].read().decode("utf-8"))
    except s3.exceptions.NoSuchKey:
        return {
            "total_feedback_count": 0,
            "pending_count": 0,
            "last_training_at": None,
            "model_version": "v6",
        }
    except Exception as e:
        logger.error(f"Failed to get metadata: {e}")
        return {}


def update_metadata(
    pending_count: int = None,
    training_triggered: bool = False,
    new_model_version: str = None,
    training_rejected: bool = False,
):
    """
    메타데이터 업데이트

    Args:
        training_triggered: 학습 성공(모델 교체). last_training_at 갱신 +
            pending_count 리셋.
        training_rejected: 품질 게이트 거부. pending 파일을 그대로 두므로
            pending_count 를 리셋하면 안 되고, training_triggered_at 만 갱신해
            2시간 쿨다운으로 트리거 폭주를 막는다
            (services/mlops/s3_feedback_store.py 의 should_trigger_training 참조).
    """
    s3 = get_s3_client()

    try:
        metadata = get_metadata()

        if training_triggered:
            metadata["last_training_at"] = datetime.now(timezone.utc).isoformat()
            metadata["pending_count"] = 0

        if training_rejected:
            metadata["training_triggered_at"] = datetime.now(timezone.utc).isoformat()
            metadata["last_rejected_at"] = metadata["training_triggered_at"]

        if pending_count is not None:
            metadata["pending_count"] = pending_count

        if new_model_version:
            metadata["model_version"] = new_model_version

        s3.put_object(
            Bucket=S3_BUCKET,
            Key="feedback/metadata.json",
            Body=json.dumps(metadata, indent=2, ensure_ascii=False),
            ContentType="application/json",
        )
        logger.info(f"✅ 메타데이터 업데이트 완료: {metadata}")

    except Exception as e:
        logger.error(f"Failed to update metadata: {e}")


def list_npz_keys(s3, prefix: str) -> List[str]:
    """
    프리픽스 하위의 .npz 키를 페이지네이터로 전부 나열한다.

    list_objects_v2는 응답당 1000개 제한이 있으므로 반드시 paginator를 사용한다.
    (feedback/processed/ 는 이미 800개 이상 존재)
    """
    keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj.get("Key", "")
            if key.endswith(".npz"):
                keys.append(key)

    return keys


def extract_analysis_id(data: Any, key: str) -> str:
    """
    NPZ 의 metadata JSON 에서 analysis_id 를 읽고, 없으면 파일명에서 추출한다.

    metadata 는 dtype=str 배열이므로 allow_pickle=False 로도 읽을 수 있다.
    """
    try:
        raw = data["metadata"]
        payload = json.loads(str(np.asarray(raw).reshape(-1)[0]))
        analysis_id = payload.get("analysis_id")
        if analysis_id:
            return str(analysis_id)
    except Exception:
        pass

    return analysis_id_from_key(key)


def load_pending_feedbacks(
    include_processed: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, List[str], List[str]]:
    """
    S3에서 피드백 데이터 로드

    Args:
        include_processed: True이면 feedback/processed/ 하위 npz도 함께 로드한다
            (전체 재학습용). 이동 대상은 여전히 pending/ 파일뿐이다.

    Returns:
        (face_features, skin_features, style_embeddings, ground_truths,
         count, file_keys, analysis_ids)
    """
    s3 = get_s3_client()

    try:
        target_keys = list_npz_keys(s3, PENDING_PREFIX)
        logger.info(f"📄 pending npz: {len(target_keys)}개")

        if include_processed:
            processed_keys = list_npz_keys(s3, PROCESSED_PREFIX)
            logger.info(
                f"📄 processed npz: {len(processed_keys)}개 (include_processed)"
            )
            target_keys = target_keys + processed_keys

        if not target_keys:
            logger.info("No pending feedbacks found")
            return None, None, None, None, 0, [], []

        face_list = []
        skin_list = []
        style_list = []
        gt_list = []
        file_keys = []
        analysis_ids = []

        for key in target_keys:
            try:
                obj_response = s3.get_object(Bucket=S3_BUCKET, Key=key)
                buffer = io.BytesIO(obj_response["Body"].read())
                # 보안: S3 NPZ는 신뢰할 수 없는 입력이므로 pickle 역직렬화 차단
                data = np.load(buffer, allow_pickle=False)

                face_list.append(data["face_features"])
                skin_list.append(data["skin_features"])
                style_list.append(data["style_embedding"])
                gt_list.append(data["ground_truth"])
                file_keys.append(key)
                analysis_ids.append(extract_analysis_id(data, key))

            except Exception as e:
                logger.warning(f"Failed to load {key}: {e}")
                continue

        if not face_list:
            return None, None, None, None, 0, [], []

        face_features = np.stack(face_list)
        skin_features = np.stack(skin_list)
        style_embeddings = np.stack(style_list)
        ground_truths = np.concatenate(gt_list)

        logger.info(f"✅ {len(face_list)}개 피드백 데이터 로드 완료")

        return (
            face_features,
            skin_features,
            style_embeddings,
            ground_truths,
            len(face_list),
            file_keys,
            analysis_ids,
        )

    except Exception as e:
        logger.error(f"❌ 피드백 데이터 로드 실패: {e}")
        return None, None, None, None, 0, [], []


def get_source_model_key(from_base: bool = False) -> str:
    """학습 시작점이 될 모델의 S3 키"""
    return BASE_MODEL_KEY if from_base else CURRENT_MODEL_KEY


def load_base_model(
    allow_random_init: bool = False,
    from_base: bool = False,
) -> Tuple[Optional[RecommendationModelV6], Dict[str, Any]]:
    """
    S3에서 학습 시작점 모델 로드

    Args:
        allow_random_init: True이면 기존 모델이 없을 때 랜덤 초기화 모델을 생성.
            기본값 False - 랜덤 초기화 모델이 운영 모델로 배포되는 것을 막는다.
            from_base=True 일 때는 무시된다 (base는 반드시 존재해야 함).
        from_base: True이면 models/base/model.pt(번들 v6 체크포인트)에서 시작한다.
            fine-tune 결과 위에 다시 fine-tune 하며 누적되는 드리프트를 끊기 위한 모드.

    Returns:
        (model, config)
    """
    s3 = get_s3_client()
    model_key = get_source_model_key(from_base)

    try:
        # S3에서 시작점 모델 다운로드
        response = s3.get_object(Bucket=S3_BUCKET, Key=model_key)

        buffer = io.BytesIO(response["Body"].read())

        # CPU에서 로드
        # 보안: S3 체크포인트는 신뢰할 수 없으므로 weights_only=True (pickle RCE 차단)
        checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)

        # 설정 추출
        # 번들 v6 체크포인트 레이아웃:
        #   {epoch, model_state_dict, optimizer_state_dict, best_val_loss, history, config}
        # config에 없는 키는 기본값으로 채운다 (구형 체크포인트 대비).
        config = dict(DEFAULT_MODEL_CONFIG)
        loaded_config = checkpoint.get("config") or {}
        if isinstance(loaded_config, dict):
            config.update(loaded_config)

        # 모델 생성 및 가중치 로드
        model = RecommendationModelV6(
            face_feat_dim=config.get("face_feat_dim", 6),
            skin_feat_dim=config.get("skin_feat_dim", 2),
            style_embed_dim=config.get("style_embed_dim", 384),
            token_dim=config.get("token_dim", 128),
            num_heads=config.get("num_heads", 4),
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        logger.info(
            f"✅ 시작점 모델 로드 완료: key={model_key}, "
            f"version={config.get('version', 'unknown')}"
        )
        return model, config

    except s3.exceptions.NoSuchKey:
        if from_base:
            logger.error(f"❌ base model missing: upload {BASE_MODEL_KEY}")
            return None, {}

        if not allow_random_init:
            logger.error(
                "❌ base model missing: upload models/current/model.pt first "
                "(랜덤 초기화 모델 배포 방지 - 강제하려면 event에 "
                '"allow_random_init": true 를 전달)'
            )
            return None, {}

        logger.warning("⚠️ 기존 모델이 없음 - 새 모델 생성 (allow_random_init=true)")
        model = RecommendationModelV6()
        return model, dict(DEFAULT_MODEL_CONFIG)

    except Exception as e:
        logger.error(f"❌ 모델 로드 실패: {e}")
        traceback.print_exc()
        return None, {}


def set_batchnorm_eval(model: nn.Module) -> int:
    """
    모든 BatchNorm 모듈을 eval 모드로 고정한다 (fine-tuning 전용).

    이유:
    - fine-tuning 샘플 수가 적어(수백 건) model.train() 상태로 학습하면
      running_mean/running_var 가 소수의 실사용자 분포로 덮어써진다.
      일부 채널의 running_var 가 0에 가까워지면 서빙 시 정규화가 폭주해
      작은 특징 편차가 크게 증폭된다.
    - eval 모드에서는 저장된 running stats 로 정규화하고 통계를 갱신하지 않는다.
      BN 의 affine 파라미터(weight/bias)는 requires_grad 를 그대로 두므로 계속 학습된다.

    Returns:
        eval 로 고정한 BatchNorm 모듈 수
    """
    frozen = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            frozen += 1
    return frozen


def fine_tune_model(
    model: RecommendationModelV6,
    face_features: np.ndarray,
    skin_features: np.ndarray,
    style_embeddings: np.ndarray,
    ground_truths: np.ndarray,
    epochs: int = FINE_TUNE_EPOCHS,
    lr: float = FINE_TUNE_LR,
) -> Tuple[RecommendationModelV6, Dict[str, Any]]:
    """
    피드백 데이터로 모델 Fine-tuning

    Returns:
        (fine_tuned_model, training_stats)
    """
    device = torch.device("cpu")  # Lambda는 CPU만 사용
    model = model.to(device)
    model.train()

    # BatchNorm running stats 고정 (드리프트 방지)
    frozen_bn = set_batchnorm_eval(model)
    logger.info(f"🧊 BatchNorm {frozen_bn}개 eval 고정 (running stats 갱신 안 함)")

    # 데이터셋 및 데이터로더
    dataset = FeedbackDataset(
        face_features, skin_features, style_embeddings, ground_truths
    )
    # BatchNorm은 batch size 1에서 실패하므로 마지막 배치를 버린다.
    # (샘플 수가 BATCH_SIZE 이하이면 전부 사용하고, 아래 루프에서 크기 1 배치는 건너뜀)
    drop_last = len(dataset) > BATCH_SIZE
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last
    )

    # 옵티마이저 및 손실 함수
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    # 학습 기록
    training_stats = {
        "epochs": epochs,
        "samples": len(dataset),
        "losses": [],
        "batchnorm_frozen": frozen_bn,
    }

    logger.info(f"🏋️ Fine-tuning 시작: {len(dataset)}개 샘플, {epochs} 에폭")

    for epoch in range(epochs):
        # 에폭마다 train 모드 복구 후 BatchNorm 만 다시 eval 로 고정
        model.train()
        set_batchnorm_eval(model)

        total_loss = 0.0
        num_batches = 0

        for face, skin, style, gt in dataloader:
            if face.size(0) < 2:
                # BatchNorm은 배치 크기 1에서 예외가 발생하므로 건너뜀
                logger.warning("⚠️ 배치 크기 1 - BatchNorm 오류 방지를 위해 건너뜀")
                continue

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
        training_stats["losses"].append(avg_loss)

        if (epoch + 1) % 2 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch + 1}/{epochs}: loss = {avg_loss:.4f}")

    model.eval()
    training_stats["final_loss"] = (
        training_stats["losses"][-1] if training_stats["losses"] else 0
    )

    logger.info(f"✅ Fine-tuning 완료: final_loss = {training_stats['final_loss']:.4f}")

    return model, training_stats


def evaluate_model(
    model: RecommendationModelV6,
    face_features: np.ndarray,
    skin_features: np.ndarray,
    style_embeddings: np.ndarray,
    ground_truths: np.ndarray,
) -> Dict[str, Any]:
    """
    모델 평가 (학습 전후 비교용)

    학습 데이터로 모델의 예측 정확도를 측정합니다.

    Args:
        model: 평가할 모델
        face_features: 얼굴 특징 배열
        skin_features: 피부 특징 배열
        style_embeddings: 스타일 임베딩 배열
        ground_truths: 정답 점수 배열 (0~1 정규화)

    Returns:
        평가 지표 딕셔너리
    """
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # 입력 스케일링 (FeedbackDataset / serving과 동일하게 맞춤)
    scaled_face, scaled_skin = scale_feature_batch(face_features, skin_features)

    # 텐서 변환
    face_tensor = torch.FloatTensor(scaled_face).to(device)
    skin_tensor = torch.FloatTensor(scaled_skin).to(device)
    style_tensor = torch.FloatTensor(style_embeddings).to(device)
    gt_tensor = torch.FloatTensor(ground_truths).reshape(-1, 1).to(device)

    with torch.no_grad():
        predictions = model(face_tensor, skin_tensor, style_tensor)

    # NumPy 변환
    preds = predictions.cpu().numpy().flatten()
    gts = gt_tensor.cpu().numpy().flatten()

    # 지표 계산
    mse = float(np.mean((preds - gts) ** 2))
    mae = float(np.mean(np.abs(preds - gts)))
    rmse = float(np.sqrt(mse))

    # 점수를 원래 범위로 역변환 (0~1 → 10~95)
    preds_original = preds * LABEL_RANGE + LABEL_MIN
    gts_original = gts * LABEL_RANGE + LABEL_MIN

    # 임계값 기반 분류 (70점 이상 = 긍정)
    threshold = (70 - LABEL_MIN) / LABEL_RANGE  # 정규화된 임계값
    pred_positive = (preds >= threshold).astype(int)
    gt_positive = (gts >= threshold).astype(int)

    # Precision, Recall, Hit Rate 계산
    true_positives = np.sum((pred_positive == 1) & (gt_positive == 1))
    false_positives = np.sum((pred_positive == 1) & (gt_positive == 0))
    false_negatives = np.sum((pred_positive == 0) & (gt_positive == 1))

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # 상관계수 (예측과 실제의 관계)
    if len(preds) > 1:
        correlation = float(np.corrcoef(preds, gts)[0, 1])
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0

    # 상위 K개 정확도 (Top-K Accuracy)
    # 예측 점수 상위 K개가 실제 긍정 피드백과 얼마나 일치하는지
    k_values = [1, 3, 5]
    top_k_accuracy = {}
    n_samples = len(preds)

    for k in k_values:
        if n_samples >= k:
            top_k_indices = np.argsort(preds)[-k:][::-1]  # 상위 K개 인덱스
            top_k_gt = gts[top_k_indices]
            top_k_accuracy[k] = float(np.mean(top_k_gt >= threshold))
        else:
            top_k_accuracy[k] = 0.0

    metrics = {
        # 회귀 지표
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        # 분류 지표 (임계값 70점 기준)
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        # 상관관계
        "correlation": correlation,
        # Top-K 정확도
        "top_k_accuracy": top_k_accuracy,
        # 통계
        "num_samples": int(n_samples),
        "avg_prediction": float(np.mean(preds_original)),
        "avg_ground_truth": float(np.mean(gts_original)),
        "std_prediction": float(np.std(preds_original)),
        "std_ground_truth": float(np.std(gts_original)),
    }

    logger.info(f"📊 모델 평가 완료:")
    logger.info(f"  MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    logger.info(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    logger.info(f"  Correlation: {correlation:.4f}")
    logger.info(f"  Top-K Accuracy: {top_k_accuracy}")

    return metrics


# ========== 홀드아웃 평가 + 품질 게이트 ==========


def predict_normalized(
    model: RecommendationModelV6,
    face_features: np.ndarray,
    skin_features: np.ndarray,
    style_embeddings: np.ndarray,
) -> np.ndarray:
    """모델 예측값(정규화 0~1 공간)을 1차원 배열로 반환한다"""
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    scaled_face, scaled_skin = scale_feature_batch(face_features, skin_features)

    face_tensor = torch.FloatTensor(scaled_face).to(device)
    skin_tensor = torch.FloatTensor(scaled_skin).to(device)
    style_tensor = torch.FloatTensor(style_embeddings).to(device)

    with torch.no_grad():
        predictions = model(face_tensor, skin_tensor, style_tensor)

    return predictions.cpu().numpy().flatten()


def pairwise_ranking_accuracy(
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    analysis_ids: List[str],
) -> Tuple[Optional[float], int]:
    """
    같은 analysis 안에서 good 이 bad 보다 높은 점수를 받은 비율.

    같은 얼굴에 대해 어떤 스타일을 더 위로 올리는지가 추천 품질의 본질이므로,
    절대 점수(MSE)만으로는 잡히지 않는 순위 붕괴를 잡아낸다.
    서로 다른 analysis 사이의 쌍은 비교 대상이 아니므로 세지 않는다.
    동점은 "더 높다"가 아니므로 오답으로 센다.

    Returns:
        (accuracy, num_pairs) - 유효한 쌍이 하나도 없으면 (None, 0)
    """
    groups: Dict[str, List[int]] = {}
    for idx, analysis_id in enumerate(analysis_ids):
        groups.setdefault(analysis_id, []).append(idx)

    correct = 0
    total = 0

    for indices in groups.values():
        if len(indices) < 2:
            continue
        for i in indices:
            for j in indices:
                if ground_truths[i] <= ground_truths[j]:
                    continue
                # i 가 더 좋은 피드백 -> 예측도 i 가 더 높아야 한다
                total += 1
                if predictions[i] > predictions[j]:
                    correct += 1

    if total == 0:
        return None, 0

    return correct / total, total


def evaluate_holdout(
    model: RecommendationModelV6,
    face_features: np.ndarray,
    skin_features: np.ndarray,
    style_embeddings: np.ndarray,
    ground_truths: np.ndarray,
    analysis_ids: List[str],
) -> Dict[str, Any]:
    """
    홀드아웃 지표 계산 (품질 게이트 전용)

    Args:
        ground_truths: 정규화된 정답 (0~1). 모델 출력과 같은 공간이어야 한다.

    Returns:
        {"mse", "mae", "ranking_accuracy", "num_pairs", "num_samples"}
    """
    preds = predict_normalized(model, face_features, skin_features, style_embeddings)
    gts = np.asarray(ground_truths, dtype=np.float64).flatten()

    mse = float(np.mean((preds - gts) ** 2))
    mae = float(np.mean(np.abs(preds - gts)))
    ranking_accuracy, num_pairs = pairwise_ranking_accuracy(preds, gts, analysis_ids)

    metrics = {
        "mse": mse,
        "mae": mae,
        "ranking_accuracy": ranking_accuracy,
        "num_pairs": num_pairs,
        "num_samples": int(len(gts)),
    }

    logger.info(
        f"📊 홀드아웃 평가: n={metrics['num_samples']} mse={mse:.5f} "
        f"mae={mae:.5f} rank_acc={ranking_accuracy} pairs={num_pairs}"
    )

    return metrics


def evaluate_quality_gate(
    before: Optional[Dict[str, Any]],
    after: Optional[Dict[str, Any]],
    holdout_size: int,
    skip_gate: bool = False,
    mse_tolerance: float = GATE_MSE_TOLERANCE,
    rank_tolerance: float = GATE_RANK_TOLERANCE,
    min_holdout: int = MIN_HOLDOUT_SAMPLES,
) -> Dict[str, Any]:
    """
    학습 전/후 홀드아웃 지표를 비교해 models/current/model.pt 교체 여부를 정한다.

    통과 조건:
    - MSE(정규화 공간) <= 이전 MSE * (1 + mse_tolerance)
    - ranking accuracy >= 이전 ranking accuracy - rank_tolerance
      (어느 한쪽이라도 유효한 쌍이 없어 None 이면 MSE 만으로 판정)

    Returns:
        {"passed", "reason", "holdout_size", "before", "after"}
    """
    gate: Dict[str, Any] = {
        "passed": False,
        "reason": "",
        "holdout_size": int(holdout_size),
        "before": before,
        "after": after,
        "mse_tolerance": mse_tolerance,
        "rank_tolerance": rank_tolerance,
    }

    if skip_gate:
        gate["passed"] = True
        gate["reason"] = "skipped"
        gate["skipped"] = True
        return gate

    if holdout_size < min_holdout:
        gate["reason"] = "insufficient_holdout"
        gate["detail"] = f"holdout {holdout_size} < {min_holdout}"
        return gate

    if not before or not after:
        gate["reason"] = "missing_metrics"
        return gate

    before_mse = float(before.get("mse", float("inf")))
    after_mse = float(after.get("mse", float("inf")))
    mse_budget = before_mse * (1.0 + mse_tolerance)

    if not (after_mse <= mse_budget):
        gate["reason"] = "mse_regressed"
        gate["detail"] = (
            f"mse {after_mse:.6f} > 허용치 {mse_budget:.6f} "
            f"(이전 {before_mse:.6f} × {1.0 + mse_tolerance:.3f})"
        )
        return gate

    before_rank = before.get("ranking_accuracy")
    after_rank = after.get("ranking_accuracy")

    if before_rank is None or after_rank is None:
        gate["passed"] = True
        gate["reason"] = "passed_mse_only"
        gate["detail"] = "유효한 pairwise 쌍이 없어 MSE 만으로 판정"
        return gate

    rank_floor = float(before_rank) - rank_tolerance
    if float(after_rank) < rank_floor:
        gate["reason"] = "ranking_regressed"
        gate["detail"] = (
            f"ranking_accuracy {float(after_rank):.4f} < 허용치 {rank_floor:.4f} "
            f"(이전 {float(before_rank):.4f} - {rank_tolerance:.3f})"
        )
        return gate

    gate["passed"] = True
    gate["reason"] = "passed"
    return gate


def save_evaluation_report(
    before_metrics: Dict[str, Any],
    after_metrics: Dict[str, Any],
    training_stats: Dict[str, Any],
    version: str,
    gate: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    평가 리포트를 S3에 저장

    Args:
        before_metrics: 학습 전 평가 지표
        after_metrics: 학습 후 평가 지표
        training_stats: 학습 통계
        version: 모델 버전
        gate: 품질 게이트 판정 결과 (거부 사유 추적용)

    Returns:
        성공 여부
    """
    s3 = get_s3_client()

    try:
        # 개선율 계산
        improvements = {}
        for key in ["mse", "mae", "rmse"]:
            if key in before_metrics and key in after_metrics:
                before_val = before_metrics[key]
                after_val = after_metrics[key]
                if before_val > 0:
                    # 손실 지표는 감소가 개선
                    improvements[key] = (before_val - after_val) / before_val * 100

        for key in ["precision", "recall", "f1_score", "correlation"]:
            if key in before_metrics and key in after_metrics:
                before_val = before_metrics[key]
                after_val = after_metrics[key]
                if before_val > 0:
                    # 정확도 지표는 증가가 개선
                    improvements[key] = (after_val - before_val) / before_val * 100

        report = {
            "version": version,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "before_training": before_metrics,
            "after_training": after_metrics,
            "improvements": improvements,
            "training_stats": training_stats,
            "gate": gate,
            "model_promoted": bool(gate.get("passed")) if gate else None,
            "summary": {
                "mse_improved": improvements.get("mse", 0) > 0,
                "precision_improved": improvements.get("precision", 0) > 0,
                "overall_improved": sum(1 for v in improvements.values() if v > 0)
                > len(improvements) / 2,
            },
        }

        # S3에 저장
        report_key = f"evaluations/{version}_report.json"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=report_key,
            Body=json.dumps(report, indent=2, ensure_ascii=False),
            ContentType="application/json",
        )

        logger.info(f"✅ 평가 리포트 저장: {report_key}")
        logger.info(f"📈 개선율: {improvements}")

        return True

    except Exception as e:
        logger.error(f"❌ 평가 리포트 저장 실패: {e}")
        traceback.print_exc()
        return False


def save_model_to_s3(
    model: RecommendationModelV6, config: Dict[str, Any], new_version: str
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
            "model_state_dict": model.state_dict(),
            "config": config,
            "version": new_version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }

        # 바이너리로 직렬화
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        model_bytes = buffer.getvalue()

        # 1. 아카이브에 백업
        archive_key = f"models/archive/{new_version}.pt"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=archive_key,
            Body=model_bytes,
            ContentType="application/octet-stream",
        )
        logger.info(f"✅ 모델 아카이브 저장: {archive_key}")

        # 2. 현재 모델 교체
        s3.put_object(
            Bucket=S3_BUCKET,
            Key="models/current/model.pt",
            Body=model_bytes,
            ContentType="application/octet-stream",
        )
        logger.info(f"✅ 현재 모델 교체 완료")

        # 3. 무결성 메타데이터 기록 (서빙 측 SHA-256 검증용)
        save_current_model_metadata(model_bytes, config, new_version)

        return True

    except Exception as e:
        logger.error(f"❌ 모델 저장 실패: {e}")
        traceback.print_exc()
        return False


def save_rejected_model(
    model: RecommendationModelV6, config: Dict[str, Any], new_version: str
) -> bool:
    """
    품질 게이트를 통과하지 못한 모델을 models/rejected/{version}.pt 에만 저장한다.

    models/current/ 와 models/archive/ 는 건드리지 않는다. 서빙은 기존 모델을
    계속 쓰고, 거부된 가중치는 사후 분석용으로만 남는다.

    Returns:
        성공 여부
    """
    s3 = get_s3_client()

    try:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "version": new_version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "rejected": True,
        }

        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)

        rejected_key = f"{REJECTED_MODEL_PREFIX}{new_version}.pt"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=rejected_key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )
        logger.warning(f"🚫 게이트 거부 모델 보관: {rejected_key}")

        return True

    except Exception as e:
        logger.error(f"❌ 거부 모델 저장 실패: {e}")
        traceback.print_exc()
        return False


def save_current_model_metadata(
    model_bytes: bytes, config: Dict[str, Any], new_version: str
) -> bool:
    """
    models/current/metadata.json 갱신 (업로드된 모델의 SHA-256 포함)

    서빙 측(models/ml_recommender.py)이 다운로드한 모델의 무결성을 검증한다.
    기존 metadata.json 이 있으면 내용을 병합한다.
    """
    s3 = get_s3_client()
    metadata_key = "models/current/metadata.json"

    try:
        metadata: Dict[str, Any] = {}
        try:
            response = s3.get_object(Bucket=S3_BUCKET, Key=metadata_key)
            existing = json.loads(response["Body"].read().decode("utf-8"))
            if isinstance(existing, dict):
                metadata = existing
        except Exception:
            metadata = {}

        metadata.update(
            {
                "sha256": hashlib.sha256(model_bytes).hexdigest(),
                "size_bytes": len(model_bytes),
                "version": new_version,
                "config": config,
                # 재학습 출처 추적 (from_base 전체 재학습 여부)
                "from_base": bool(config.get("from_base", False)),
                "include_processed": bool(config.get("include_processed", False)),
                "source_model_key": config.get("source_model_key"),
                "batchnorm_frozen": bool(config.get("batchnorm_frozen", True)),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        s3.put_object(
            Bucket=S3_BUCKET,
            Key=metadata_key,
            Body=json.dumps(metadata, indent=2, ensure_ascii=False),
            ContentType="application/json",
        )
        logger.info(
            f"✅ 모델 메타데이터 저장: {metadata_key} (sha256={metadata['sha256'][:12]}...)"
        )
        return True

    except Exception as e:
        logger.error(f"❌ 모델 메타데이터 저장 실패: {e}")
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
            # include_processed 모드에서 섞여 들어온 processed/ 키는 이동하지 않는다
            if not old_key.startswith(PENDING_PREFIX):
                continue

            filename = old_key.split("/")[-1]
            new_key = f"feedback/processed/{batch_name}/{filename}"

            # Copy then delete
            s3.copy_object(
                Bucket=S3_BUCKET,
                CopySource={"Bucket": S3_BUCKET, "Key": old_key},
                Key=new_key,
            )
            s3.delete_object(Bucket=S3_BUCKET, Key=old_key)
            moved_count += 1

        logger.info(f"✅ {moved_count}개 파일을 processed로 이동: {batch_name}")
        return True

    except Exception as e:
        logger.error(f"❌ 파일 이동 실패: {e}")
        return False


def run_training_pipeline(event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    전체 학습 파이프라인 실행

    Args:
        event: Lambda 이벤트
            - allow_random_init: 기존 모델이 없을 때 랜덤 초기화 허용
            - force: MIN_SAMPLES 게이트 우회 (품질 게이트는 우회하지 않는다)
            - skip_gate: 품질 게이트 우회 (from_base 전체 재학습처럼 의도적일 때만)
            - from_base: models/base/model.pt 에서 시작 (드리프트 누적 차단)
            - include_processed: feedback/processed/ 데이터도 학습에 포함

    Returns:
        결과 딕셔너리
    """
    event = event or {}
    allow_random_init = bool(event.get("allow_random_init", False))
    force_train = bool(event.get("force", False))
    # force 는 MIN_SAMPLES 만 우회한다. 품질 게이트 우회는 별도 플래그.
    skip_gate = bool(event.get("skip_gate", False))
    from_base = bool(event.get("from_base", False))
    include_processed = bool(event.get("include_processed", False))
    source_model_key = get_source_model_key(from_base)
    timestamp = datetime.now(timezone.utc)
    date_str = timestamp.strftime("%Y%m%d")
    new_version = f"v6_feedback_{date_str}"
    batch_name = f'batch_{timestamp.strftime("%Y%m%d_%H%M%S")}'

    result = {
        "success": False,
        "new_version": new_version,
        "samples_trained": 0,
        "final_loss": None,
        "steps_completed": [],
        "from_base": from_base,
        "include_processed": include_processed,
        "source_model_key": source_model_key,
        "skip_gate": skip_gate,
        "model_promoted": False,
        "gate": None,
    }

    try:
        # 1. 피드백 데이터 로드
        logger.info(
            f"📥 Step 1: 피드백 데이터 로드 (include_processed={include_processed})"
        )
        face, skin, style, gt, count, file_keys, analysis_ids = load_pending_feedbacks(
            include_processed=include_processed
        )

        if count == 0:
            result["message"] = "No pending feedbacks"
            return result

        # 메타데이터의 pending_count가 아니라 "실제로 로드된 샘플 수"로 검증한다.
        # (카운터가 드리프트해도 9개 파일로 학습이 시작되지 않도록)
        # force=True 이벤트는 기존 동작대로 이 검사를 우회한다.
        if not force_train and count < MIN_SAMPLES:
            message = f"Insufficient data: {count}/{MIN_SAMPLES} loaded samples"
            logger.info(f"⏸️ {message}")
            result["message"] = message
            result["loaded_count"] = count
            return result

        result["samples_trained"] = count
        result["loaded_count"] = count
        result["steps_completed"].append("load_feedbacks")

        # 평가용 정답 라벨 정규화 (10~95 -> 0~1)
        # 모델 출력이 sigmoid(0~1)이므로 동일 공간에서 비교해야 한다.
        gt_normalized = (gt - LABEL_MIN) / LABEL_RANGE

        # 1.5. 학습/홀드아웃 결정적 분할 (analysis_id 해시)
        logger.info(f"✂️ Step 1.5: 홀드아웃 분할 (ratio={TRAIN_HOLDOUT_RATIO})")
        train_idx, holdout_idx = split_holdout_indices(
            analysis_ids, TRAIN_HOLDOUT_RATIO
        )
        result["train_size"] = len(train_idx)
        result["holdout_size"] = len(holdout_idx)
        result["holdout_ratio"] = TRAIN_HOLDOUT_RATIO
        logger.info(
            f"  학습 {len(train_idx)}건 / 홀드아웃 {len(holdout_idx)}건 "
            f"(총 {count}건)"
        )

        if not train_idx:
            result["message"] = "No training samples after holdout split"
            return result

        train_face, train_skin, train_style, train_gt = take_indices(
            (face, skin, style, gt), train_idx
        )

        if holdout_idx:
            hold_face, hold_skin, hold_style, hold_gt_norm = take_indices(
                (face, skin, style, gt_normalized), holdout_idx
            )
            hold_analysis_ids = [analysis_ids[i] for i in holdout_idx]
        else:
            hold_face = hold_skin = hold_style = hold_gt_norm = None
            hold_analysis_ids = []

        result["steps_completed"].append("holdout_split")

        # 2. 시작점 모델 로드 (from_base=True 이면 models/base/model.pt)
        logger.info(f"📥 Step 2: 시작점 모델 로드 (key={source_model_key})")
        model, config = load_base_model(
            allow_random_init=allow_random_init, from_base=from_base
        )

        if model is None:
            if from_base:
                message = f"base model missing: upload {BASE_MODEL_KEY}"
            elif not allow_random_init:
                message = "base model missing: upload models/current/model.pt first"
            else:
                message = "Failed to load base model"
            result["message"] = message
            return result

        result["steps_completed"].append("load_model")

        # 2.5. 학습 전 홀드아웃 평가 (현재 운영 모델의 기준선)
        before_metrics = None
        if holdout_idx:
            logger.info("📊 Step 2.5: 학습 전 홀드아웃 평가")
            before_metrics = evaluate_holdout(
                model,
                hold_face,
                hold_skin,
                hold_style,
                hold_gt_norm,
                hold_analysis_ids,
            )
            result["before_metrics"] = before_metrics
            result["steps_completed"].append("evaluate_before")

        # 3. Fine-tuning (학습 분할만 사용 - 홀드아웃 누수 방지)
        logger.info("🏋️ Step 3: Fine-tuning")
        model, stats = fine_tune_model(
            model, train_face, train_skin, train_style, train_gt
        )
        result["final_loss"] = stats["final_loss"]
        result["steps_completed"].append("fine_tune")

        # 3.5. 학습 후 홀드아웃 평가
        after_metrics = None
        if holdout_idx:
            logger.info("📊 Step 3.5: 학습 후 홀드아웃 평가")
            after_metrics = evaluate_holdout(
                model,
                hold_face,
                hold_skin,
                hold_style,
                hold_gt_norm,
                hold_analysis_ids,
            )
            result["after_metrics"] = after_metrics
            result["steps_completed"].append("evaluate_after")

        # 4. 설정 업데이트
        config["version"] = new_version
        config["fine_tuned_at"] = timestamp.isoformat()
        config["samples_count"] = count
        config["train_samples"] = len(train_idx)
        config["holdout_samples"] = len(holdout_idx)
        config["from_base"] = from_base
        config["include_processed"] = include_processed
        config["source_model_key"] = source_model_key
        # BatchNorm running stats 는 fine-tuning 중 고정된다
        config["batchnorm_frozen"] = True

        # 4.5. 품질 게이트 판정
        logger.info("🚦 Step 4.5: 품질 게이트 판정")
        gate = evaluate_quality_gate(
            before_metrics,
            after_metrics,
            holdout_size=len(holdout_idx),
            skip_gate=skip_gate,
        )
        result["gate"] = gate
        result["steps_completed"].append("quality_gate")

        if not gate["passed"]:
            # 거부: 현재 모델 유지, 가중치는 rejected/ 에만 보관
            logger.warning(
                f"🚫 품질 게이트 거부 ({gate['reason']}): "
                f"{gate.get('detail', '')} - models/current 유지"
            )
            config["gate_rejected"] = True
            config["gate_reason"] = gate["reason"]
            save_rejected_model(model, config, new_version)
            result["steps_completed"].append("save_rejected_model")

            # pending 은 이동하지 않는다 (다음 학습에 재포함)
            result["pending_files_moved"] = 0

            # pending_count 는 리셋하지 않고 training_triggered_at 만 갱신해
            # 임계값 초과 상태에서 트리거가 매번 재발동하는 것을 막는다.
            update_metadata(training_rejected=True)
            result["steps_completed"].append("update_metadata")

            save_evaluation_report(
                before_metrics, after_metrics, stats, new_version, gate=gate
            )
            result["steps_completed"].append("save_evaluation_report")

            result["success"] = False
            result["model_promoted"] = False
            result["message"] = f"Quality gate rejected: {gate['reason']}"
            return result

        # 5. 모델 저장 (게이트 통과분만 models/current 교체)
        logger.info("💾 Step 5: 모델 저장")
        if not save_model_to_s3(model, config, new_version):
            result["message"] = "Failed to save model"
            return result

        result["model_promoted"] = True
        result["steps_completed"].append("save_model")

        # 6. pending → processed 이동 (processed/ 에서 읽어온 파일은 그대로 둔다)
        logger.info("📦 Step 6: 피드백 파일 이동")
        pending_keys = [k for k in file_keys if k.startswith(PENDING_PREFIX)]
        result["pending_files_moved"] = len(pending_keys)
        move_pending_to_processed(pending_keys, batch_name)
        result["steps_completed"].append("move_feedbacks")

        # 7. 메타데이터 업데이트
        logger.info("📝 Step 7: 메타데이터 업데이트")
        update_metadata(training_triggered=True, new_model_version=new_version)
        result["steps_completed"].append("update_metadata")

        # 8. 평가 리포트 저장
        logger.info("📊 Step 8: 평가 리포트 저장")
        save_evaluation_report(
            before_metrics, after_metrics, stats, new_version, gate=gate
        )
        result["steps_completed"].append("save_evaluation_report")

        result["success"] = True
        result["message"] = "Training completed successfully"

        logger.info(f"✅ 학습 파이프라인 완료: {new_version}")

        return result

    except Exception as e:
        logger.error(f"❌ 학습 파이프라인 실패: {e}")
        traceback.print_exc()
        result["message"] = str(e)
        return result


def lambda_handler(event, context):
    """
    Lambda 핸들러

    Args:
        event: {
            "trigger_type": "scheduled" | "data_threshold" | "manual",
            "force": false,             # true이면 MIN_SAMPLES 무시 (게이트는 유지)
            "skip_gate": false,         # true이면 품질 게이트 우회
            "from_base": false,         # true이면 models/base/model.pt 에서 재학습
            "include_processed": false, # true이면 feedback/processed/ 도 학습에 포함
            "allow_random_init": false,
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

    trigger_type = event.get("trigger_type", "unknown")
    force_train = event.get("force", False)
    skip_gate = bool(event.get("skip_gate", False))
    from_base = bool(event.get("from_base", False))
    include_processed = bool(event.get("include_processed", False))
    timestamp = datetime.now(timezone.utc).isoformat()

    # Pending 피드백 수 확인
    pending_count = get_pending_count()
    logger.info(f"📊 Pending feedback count: {pending_count}")

    # 최소 샘플 수 1차 확인 (force가 아닐 때)
    # 주의: 여기서 쓰는 pending_count는 metadata.json 카운터이므로 드리프트할 수 있다.
    # 실제 게이트는 run_training_pipeline에서 "로드된 샘플 수"로 다시 검사한다.
    if not force_train and pending_count < MIN_SAMPLES:
        message = f"Insufficient data: {pending_count}/{MIN_SAMPLES} samples"
        logger.info(f"⏸️ {message}")

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "success": False,
                    "message": message,
                    "trigger_type": trigger_type,
                    "pending_count": pending_count,
                    "min_samples": MIN_SAMPLES,
                    "timestamp": timestamp,
                }
            ),
        }

    # 실제 학습 파이프라인 실행
    logger.info(
        f"🏋️ Training triggered with {pending_count} samples "
        f"(force={force_train}, skip_gate={skip_gate}, from_base={from_base}, "
        f"include_processed={include_processed})"
    )

    try:
        training_result = run_training_pipeline(event)

        if training_result["success"]:
            logger.info(f"✅ 학습 완료: {training_result['new_version']}")
            return {
                "statusCode": 200,
                "body": json.dumps(
                    {
                        "success": True,
                        "message": "Training completed successfully",
                        "trigger_type": trigger_type,
                        "pending_count": pending_count,
                        "from_base": from_base,
                        "include_processed": include_processed,
                        "skip_gate": skip_gate,
                        "source_model_key": training_result.get("source_model_key"),
                        "gate": training_result.get("gate"),
                        "model_promoted": training_result.get("model_promoted"),
                        "training_result": training_result,
                        "timestamp": timestamp,
                    }
                ),
            }
        else:
            logger.error(f"❌ 학습 실패: {training_result.get('message')}")
            return {
                "statusCode": 200,  # Lambda 자체는 성공, 학습만 실패
                "body": json.dumps(
                    {
                        "success": False,
                        "message": training_result.get("message", "Training failed"),
                        "trigger_type": trigger_type,
                        "pending_count": pending_count,
                        "from_base": from_base,
                        "include_processed": include_processed,
                        "skip_gate": skip_gate,
                        "source_model_key": training_result.get("source_model_key"),
                        "gate": training_result.get("gate"),
                        "model_promoted": training_result.get("model_promoted"),
                        "training_result": training_result,
                        "timestamp": timestamp,
                    }
                ),
            }

    except Exception as e:
        logger.error(f"❌ Lambda 실행 중 예외 발생: {e}")
        traceback.print_exc()

        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "success": False,
                    "message": str(e),
                    "trigger_type": trigger_type,
                    "pending_count": pending_count,
                    "timestamp": timestamp,
                }
            ),
        }
