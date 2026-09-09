#!/usr/bin/env python3
"""
DynamoDB -> S3 피드백 백필 스크립트

배경:
    DynamoDB(hairme-analysis)에는 style_{k}_feedback 이 1,000건 이상 쌓여 있지만
    MLOps 학습 데이터(S3: hairme-mlops/feedback/pending/)에는 일부만 반영되어 있다.
    (MLOPS_ENABLED=false 기간 / S3 저장 실패 등으로 유실)

    이 스크립트는 DynamoDB 원본을 스캔해서
    services/mlops/s3_feedback_store.py::save_feedback 과 **동일한 NPZ 레이아웃**으로
    누락된 피드백을 S3에 백필한다.

주의:
    - save_feedback() 자체는 호출하지 않는다.
      (파일마다 metadata.json 을 갱신하고 재학습 Lambda 를 트리거할 수 있기 때문)
    - 기본 동작은 --dry-run 이다. 실제 쓰기는 --execute 를 명시해야 한다.
    - --execute 는 MLOPS_S3_BUCKET 환경변수 또는 --bucket 인자를 요구한다.

사용법:
    # 1) 미리보기 (S3/DynamoDB 읽기만 수행)
    python scripts/backfill_feedback_to_s3.py --dry-run

    # 2) 실제 업로드
    MLOPS_S3_BUCKET=hairme-mlops python scripts/backfill_feedback_to_s3.py --execute

Author: HairMe ML Team
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# 프로젝트 루트를 import 경로에 추가 (scripts/ 하위에서 실행되므로)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import services.mlops.s3_feedback_store as sfs  # noqa: E402

logger = logging.getLogger("backfill_feedback")

# ========== 상수 ==========
DEFAULT_BUCKET = "hairme-mlops"
DEFAULT_TABLE = "hairme-analysis"
DEFAULT_REGION = "ap-northeast-2"

S3_PENDING_PREFIX = sfs.S3_FEEDBACK_PREFIX  # "feedback/pending/"
S3_PROCESSED_PREFIX = sfs.S3_PROCESSED_PREFIX  # "feedback/processed/"
S3_METADATA_KEY = sfs.S3_METADATA_KEY  # "feedback/metadata.json"

STYLE_INDEXES = (1, 2, 3, 4, 5)

# save_feedback 과 동일한 라벨 매핑 (good -> 90.0 / bad -> 10.0)
GROUND_TRUTH_GOOD = 90.0
GROUND_TRUTH_BAD = 10.0

POSITIVE_FEEDBACKS = {"good", "like"}
NEGATIVE_FEEDBACKS = {"bad", "dislike"}

# database/dynamodb_connection.py 와 동일한 순서
FACE_FEATURE_FIELDS: Tuple[str, ...] = (
    "mediapipe_face_ratio",
    "mediapipe_forehead_width",
    "mediapipe_cheekbone_width",
    "mediapipe_jaw_width",
    "mediapipe_forehead_ratio",
    "mediapipe_jaw_ratio",
)
SKIN_FEATURE_FIELDS: Tuple[str, ...] = (
    "mediapipe_ITA_value",
    "mediapipe_hue_value",
)

PROJECTION_FIELDS: Tuple[str, ...] = (
    ("analysis_id", "created_at", "face_shape", "personal_color", "recommended_styles")
    + tuple(f"style_{k}_feedback" for k in STYLE_INDEXES)
    + FACE_FEATURE_FIELDS
    + SKIN_FEATURE_FIELDS
)


# ========== 값 변환 유틸 ==========
def to_float(value: Any) -> Optional[float]:
    """DynamoDB Decimal / 문자열 값을 float 로 변환 (실패 시 None)"""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def normalize_feedback(value: Any) -> Optional[str]:
    """
    피드백 값을 store 가 쓰는 'good' / 'bad' 로 정규화

    like/good -> good (ground_truth 90.0)
    dislike/bad -> bad (ground_truth 10.0)
    그 외 -> None (스킵)
    """
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in POSITIVE_FEEDBACKS:
        return "good"
    if normalized in NEGATIVE_FEEDBACKS:
        return "bad"
    return None


def ground_truth_for(feedback: str) -> float:
    """save_feedback 과 동일한 라벨 값"""
    return GROUND_TRUTH_GOOD if feedback == "good" else GROUND_TRUTH_BAD


def parse_created_at(value: Any) -> Optional[datetime]:
    """created_at(ISO8601) 파싱 - 실패 시 None"""
    if not isinstance(value, str) or not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def extract_features(item: Dict[str, Any]) -> Optional[Tuple[List[float], List[float]]]:
    """
    mediapipe 8개 필드가 모두 존재할 때만 (face 6차원, skin 2차원) 반환

    순서는 database/dynamodb_connection.py 와 동일하다.
    """
    face: List[float] = []
    for field in FACE_FEATURE_FIELDS:
        value = to_float(item.get(field))
        if value is None:
            return None
        face.append(value)

    skin: List[float] = []
    for field in SKIN_FEATURE_FIELDS:
        value = to_float(item.get(field))
        if value is None:
            return None
        skin.append(value)

    return face, skin


def extract_hairstyle_id(item: Dict[str, Any], style_index: int) -> Optional[int]:
    """recommended_styles[k-1].hairstyle_id (트렌드 스타일은 None -> 스킵)"""
    styles = item.get("recommended_styles") or []
    if not isinstance(styles, (list, tuple)):
        return None
    if style_index > len(styles):
        return None

    style_data = styles[style_index - 1]
    if not isinstance(style_data, dict):
        return None

    raw_id = style_data.get("hairstyle_id")
    if raw_id is None:
        return None
    try:
        return int(raw_id)
    except (TypeError, ValueError):
        return None


# ========== NPZ 빌더 (save_feedback 과 동일 레이아웃) ==========
def build_npz_bytes(
    analysis_id: str,
    face_shape: str,
    skin_tone: str,
    hairstyle_id: int,
    feedback: str,
    face_features: Sequence[float],
    skin_features: Sequence[float],
    style_embedding: np.ndarray,
    timestamp: datetime,
) -> bytes:
    """
    S3FeedbackStore.save_feedback 과 동일한 NPZ 바이트 생성

    keys: face_features(6, f4) / skin_features(2, f4) /
          style_embedding(384, f4) / ground_truth(1, f4) / metadata(1, <U)
    """
    npz_data = {
        "face_features": np.array(face_features, dtype=np.float32),
        "skin_features": np.array(skin_features, dtype=np.float32),
        "style_embedding": np.asarray(style_embedding).astype(np.float32),
        "ground_truth": np.array([ground_truth_for(feedback)], dtype=np.float32),
        "metadata": np.array(
            [
                json.dumps(
                    {
                        "analysis_id": analysis_id,
                        "face_shape": face_shape,
                        "skin_tone": skin_tone,
                        "hairstyle_id": hairstyle_id,
                        "feedback": feedback,
                        "timestamp": timestamp.isoformat(),
                        "source": "backfill",
                    }
                )
            ],
            dtype=str,
        ),
    }

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **npz_data)
    buffer.seek(0)
    return buffer.getvalue()


def build_filename(analysis_id: str, hairstyle_id: int, when: datetime) -> str:
    """
    store 의 신규 네이밍 규칙 + backfill_ 접두사

    backfill_{YYYY-MM-DD}_{analysis_id[:8]}_s{style_token}_{uuid6}.npz
    """
    style_token = sfs.S3FeedbackStore._style_token(hairstyle_id)
    return (
        f"backfill_{when.strftime('%Y-%m-%d')}_{analysis_id[:8]}"
        f"_s{style_token}_{uuid.uuid4().hex[:6]}.npz"
    )


# ========== 중복 판정 ==========
def is_already_present(
    existing_keys: Iterable[str], analysis_id: str, hairstyle_id: int
) -> bool:
    """
    이미 S3 에 존재하는 피드백인지 판정

    - `_{analysis_id[:8]}_s{token}_` 을 포함하는 키가 있으면 해당 스타일은 이미 존재
    - `_{analysis_id[:8]}` 은 포함하지만 스타일 접미사가 없는 구버전 키가 있으면
      어떤 스타일인지 알 수 없으므로 해당 analysis 전체를 스킵한다
    """
    marker = f"_{analysis_id[:8]}"
    style_token = sfs.S3FeedbackStore._style_token(hairstyle_id)

    legacy_pattern = re.compile(re.escape(marker) + r"_s[^_/]+_")
    exact_pattern = re.compile(
        re.escape(marker) + r"_s" + re.escape(style_token) + r"_"
    )

    for key in existing_keys:
        if marker not in key:
            continue
        if legacy_pattern.search(key):
            # 스타일 단위로 구분 가능한 신규 파일
            if exact_pattern.search(key):
                return True
            continue
        # 스타일 접미사가 없는 구버전 파일 -> analysis 전체 스킵
        return True

    return False


# ========== AWS 헬퍼 ==========
def get_s3_client(region: str):
    import boto3

    return boto3.client("s3", region_name=region)


def get_dynamodb_table(table_name: str, region: str):
    import boto3

    return boto3.resource("dynamodb", region_name=region).Table(table_name)


def list_keys(s3_client, bucket: str, prefix: str) -> List[str]:
    """prefix 아래 모든 키 목록 (페이지네이션)"""
    keys: List[str] = []
    token: Optional[str] = None

    while True:
        kwargs: Dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token

        response = s3_client.list_objects_v2(**kwargs)
        for obj in response.get("Contents", []) or []:
            keys.append(obj["Key"])

        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")
        if not token:
            break

    return keys


def count_npz_objects(s3_client, bucket: str, prefix: str) -> int:
    """prefix 아래 .npz 객체 수"""
    return sum(
        1 for key in list_keys(s3_client, bucket, prefix) if key.endswith(".npz")
    )


def scan_analyses(table, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    DynamoDB 전체 스캔 (ProjectionExpression 으로 필요한 필드만 조회)

    Args:
        table: boto3 DynamoDB Table 리소스
        limit: 최대 조회 행 수 (테스트용)
    """
    name_map = {f"#f{i}": field for i, field in enumerate(PROJECTION_FIELDS)}
    projection = ", ".join(name_map.keys())

    items: List[Dict[str, Any]] = []
    start_key: Optional[Dict[str, Any]] = None

    while True:
        kwargs: Dict[str, Any] = {
            "ProjectionExpression": projection,
            "ExpressionAttributeNames": name_map,
        }
        if start_key:
            kwargs["ExclusiveStartKey"] = start_key

        response = table.scan(**kwargs)
        items.extend(response.get("Items", []) or [])

        if limit is not None and len(items) >= limit:
            return items[:limit]

        start_key = response.get("LastEvaluatedKey")
        if not start_key:
            break

    return items


# ========== 샘플 생성 ==========
def build_samples(
    items: Sequence[Dict[str, Any]],
    existing_keys: Sequence[str],
    style_lookup,
) -> Tuple[List[Dict[str, Any]], Counter]:
    """
    DynamoDB 행 목록 -> 업로드할 샘플 목록

    Returns:
        (samples, stats)
        samples: {key, body_args...} 형태의 dict 목록
        stats: 스킵 사유별 카운터
    """
    stats: Counter = Counter()
    samples: List[Dict[str, Any]] = []

    for item in items:
        stats["rows_scanned"] += 1

        analysis_id = item.get("analysis_id")
        if not analysis_id:
            stats["skip_no_analysis_id"] += 1
            continue

        feedbacks = {
            k: normalize_feedback(item.get(f"style_{k}_feedback"))
            for k in STYLE_INDEXES
        }
        if not any(feedbacks.values()):
            stats["skip_no_feedback"] += 1
            continue

        stats["rows_with_feedback"] += 1

        features = extract_features(item)
        if features is None:
            stats["skip_missing_mediapipe"] += 1
            continue
        face_features, skin_features = features

        created_at = parse_created_at(item.get("created_at"))
        when = created_at or datetime.now(timezone.utc)

        for style_index in STYLE_INDEXES:
            feedback = feedbacks[style_index]
            if feedback is None:
                continue

            stats["feedback_values"] += 1

            hairstyle_id = extract_hairstyle_id(item, style_index)
            if hairstyle_id is None:
                # 트렌드 스타일(hairstyle_id 없음) 또는 recommended_styles 누락
                stats["skip_no_hairstyle_id"] += 1
                continue

            style_embedding = style_lookup.get_style_embedding(hairstyle_id)
            if style_embedding is None:
                stats["skip_no_embedding"] += 1
                continue

            stats["usable_samples"] += 1

            if is_already_present(existing_keys, str(analysis_id), hairstyle_id):
                stats["already_present"] += 1
                continue

            filename = build_filename(str(analysis_id), hairstyle_id, when)
            samples.append(
                {
                    "key": f"{S3_PENDING_PREFIX}{filename}",
                    "analysis_id": str(analysis_id),
                    "style_index": style_index,
                    "hairstyle_id": hairstyle_id,
                    "feedback": feedback,
                    "face_shape": item.get("face_shape") or "",
                    "skin_tone": item.get("personal_color") or "",
                    "face_features": face_features,
                    "skin_features": skin_features,
                    "style_embedding": style_embedding,
                    "created_at": when,
                    "month": when.strftime("%Y-%m"),
                }
            )
            stats["would_write"] += 1
            stats[f"label_{feedback}"] += 1

    return samples, stats


def load_style_lookup(embeddings_path: Optional[str] = None):
    """
    스타일 임베딩 조회기 생성

    S3FeedbackStore 의 임베딩 로딩/조회 로직을 그대로 재사용하되,
    __init__ (S3 클라이언트 생성) 은 건너뛴다.
    """
    if embeddings_path:
        sfs.LOCAL_EMBEDDINGS_PATH = embeddings_path

    store = sfs.S3FeedbackStore.__new__(sfs.S3FeedbackStore)
    store.bucket_name = ""
    store.region = ""
    store.s3_client = None
    store.enabled = False
    store.style_embeddings = None
    store.style_to_idx = {}
    store._load_style_embeddings()

    if store.style_embeddings is None:
        raise RuntimeError(
            f"스타일 임베딩을 로드하지 못했습니다: {sfs.LOCAL_EMBEDDINGS_PATH} "
            "(--embeddings 로 경로를 지정하세요)"
        )
    return store


# ========== 업로드 / 메타데이터 ==========
def upload_samples(s3_client, bucket: str, samples: Sequence[Dict[str, Any]]) -> int:
    """샘플들을 feedback/pending/ 에 업로드 (성공 건수 반환)"""
    uploaded = 0

    for sample in samples:
        body = build_npz_bytes(
            analysis_id=sample["analysis_id"],
            face_shape=sample["face_shape"],
            skin_tone=sample["skin_tone"],
            hairstyle_id=sample["hairstyle_id"],
            feedback=sample["feedback"],
            face_features=sample["face_features"],
            skin_features=sample["skin_features"],
            style_embedding=sample["style_embedding"],
            timestamp=sample["created_at"],
        )

        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=sample["key"],
                Body=body,
                ContentType="application/octet-stream",
            )
            uploaded += 1
        except Exception as exc:  # pragma: no cover - 네트워크 예외 경로
            logger.error(f"업로드 실패: {sample['key']} ({exc})")

    return uploaded


def rewrite_metadata(s3_client, bucket: str) -> Dict[str, Any]:
    """
    metadata.json 의 카운터를 실제 S3 객체 수로 재계산

    - pending_count = feedback/pending/ 의 .npz 개수
    - total_feedback_count = pending + processed 개수
    - 그 외 키(last_training_at, model_version, training_triggered_at 등)는 보존
      (재학습을 유발하는 필드는 건드리지 않는다)
    """
    pending = count_npz_objects(s3_client, bucket, S3_PENDING_PREFIX)
    processed = count_npz_objects(s3_client, bucket, S3_PROCESSED_PREFIX)

    metadata: Dict[str, Any] = {}
    try:
        response = s3_client.get_object(Bucket=bucket, Key=S3_METADATA_KEY)
        metadata = json.loads(response["Body"].read().decode("utf-8"))
        if not isinstance(metadata, dict):
            metadata = {}
    except Exception as exc:
        logger.warning(f"기존 metadata.json 을 읽지 못함 - 새로 생성합니다 ({exc})")
        metadata = {}

    metadata["pending_count"] = pending
    metadata["total_feedback_count"] = pending + processed
    metadata["last_backfill_at"] = datetime.now(timezone.utc).isoformat()

    s3_client.put_object(
        Bucket=bucket,
        Key=S3_METADATA_KEY,
        Body=json.dumps(metadata, ensure_ascii=False, indent=2),
        ContentType="application/json",
    )

    logger.info(
        f"metadata.json 갱신: pending_count={pending}, "
        f"total_feedback_count={pending + processed}"
    )
    return metadata


# ========== 리포트 ==========
def print_summary(
    samples: Sequence[Dict[str, Any]], stats: Counter, executed: bool
) -> None:
    months = Counter(sample["month"] for sample in samples)

    logger.info("=" * 60)
    logger.info("백필 요약 (%s)", "EXECUTE" if executed else "DRY-RUN")
    logger.info("=" * 60)
    logger.info("스캔한 행 수            : %d", stats["rows_scanned"])
    logger.info("피드백 있는 행          : %d", stats["rows_with_feedback"])
    logger.info("피드백 값 개수          : %d", stats["feedback_values"])
    logger.info("사용 가능한 샘플        : %d", stats["usable_samples"])
    logger.info("이미 S3에 존재          : %d", stats["already_present"])
    logger.info("업로드 대상(would_write): %d", stats["would_write"])
    logger.info("  - good(90.0)          : %d", stats["label_good"])
    logger.info("  - bad(10.0)           : %d", stats["label_bad"])
    logger.info("스킵: mediapipe 누락    : %d", stats["skip_missing_mediapipe"])
    logger.info("스킵: hairstyle_id 없음 : %d", stats["skip_no_hairstyle_id"])
    logger.info("스킵: 임베딩 없음       : %d", stats["skip_no_embedding"])
    logger.info("스킵: 피드백 없음(행)   : %d", stats["skip_no_feedback"])
    logger.info("-" * 60)
    logger.info("월별 분포:")
    for month in sorted(months):
        logger.info("  %s : %d", month, months[month])
    logger.info("=" * 60)


# ========== CLI ==========
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backfill_feedback_to_s3.py",
        description=(
            "DynamoDB(hairme-analysis)의 스타일 피드백을 "
            "S3(feedback/pending/)에 NPZ 로 백필합니다."
        ),
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help=f"S3 버킷 이름 (기본: 환경변수 MLOPS_S3_BUCKET 또는 {DEFAULT_BUCKET})",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help=f"DynamoDB 테이블 (기본: {DEFAULT_TABLE})",
    )
    parser.add_argument(
        "--region", default=DEFAULT_REGION, help=f"AWS 리전 (기본: {DEFAULT_REGION})"
    )
    parser.add_argument(
        "--embeddings",
        default=str(PROJECT_ROOT / "data_source" / "style_embeddings.npz"),
        help="스타일 임베딩 NPZ 경로",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="스캔할 최대 행 수 (테스트용)"
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="기본값. 읽기만 수행하고 S3에 아무것도 쓰지 않습니다.",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="실제 S3 업로드 + metadata.json 카운터 재계산 (버킷 지정 필수)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    execute = bool(args.execute)
    env_bucket = os.getenv("MLOPS_S3_BUCKET")
    bucket = args.bucket or env_bucket

    if execute and not bucket:
        logger.error(
            "--execute 를 사용하려면 --bucket 인자 또는 MLOPS_S3_BUCKET 환경변수가 "
            "필요합니다 (안전 가드)"
        )
        return 2

    bucket = bucket or DEFAULT_BUCKET

    logger.info(
        "모드=%s | bucket=%s | table=%s | region=%s | limit=%s",
        "EXECUTE" if execute else "DRY-RUN",
        bucket,
        args.table,
        args.region,
        args.limit,
    )

    style_lookup = load_style_lookup(args.embeddings)

    s3_client = get_s3_client(args.region)
    existing_keys = list_keys(s3_client, bucket, S3_PENDING_PREFIX) + list_keys(
        s3_client, bucket, S3_PROCESSED_PREFIX
    )
    logger.info("기존 S3 피드백 객체: %d개", len(existing_keys))

    table = get_dynamodb_table(args.table, args.region)
    items = scan_analyses(table, limit=args.limit)
    logger.info("DynamoDB 스캔 완료: %d행", len(items))

    samples, stats = build_samples(items, existing_keys, style_lookup)

    if not execute:
        print_summary(samples, stats, executed=False)
        logger.info("DRY-RUN: S3에 아무것도 쓰지 않았습니다. (--execute 로 실행)")
        return 0

    uploaded = upload_samples(s3_client, bucket, samples)
    logger.info("업로드 완료: %d/%d", uploaded, len(samples))

    rewrite_metadata(s3_client, bucket)
    print_summary(samples, stats, executed=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
