"""hairme-analysis 항목에서 학습 샘플 입력을 꺼내는 공통 함수

학습 샘플을 쓰는 모든 경로가 같은 규칙을 쓰도록 모아 둔다:
    - database/dynamodb_connection._save_feedback_to_s3 (명시 피드백 실시간 저장)
    - scripts/backfill_feedback_to_s3.py (명시 피드백 백필)
    - services/mlops/implicit_signals.py (암묵 피드백)
측정값 순서는 database/dynamodb_connection.py 의 저장 순서와 같다.

측정값이 하나라도 없으면 샘플을 만들지 않는다. 기본값(400/500/0.8 ...)이나 얼굴형
one-hot 으로 채우면 트레이너가 그것을 픽셀 측정값으로 학습한다.
"""

import math
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple

from models.mediapipe_analyzer import (
    FEATURE_VERSION_3D,
    FEATURE_VERSION_LEGACY,
)

# 분석 행의 측정 방식 표식 필드 (database/dynamodb_repository.py 가 저장)
FEATURE_VERSION_FIELD = "mediapipe_feature_version"
# 표식 이전(f5b170e 배포 ~ 이 필드 추가 전)의 3D 행 판별용: 3D 측정과 함께 추가된 필드
_HEAD_POSE_FIELD = "mediapipe_head_yaw"

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


def _finite_float(value: Any) -> Optional[float]:
    number = to_float(value)
    if number is None or not math.isfinite(number):
        return None
    return number


def missing_feature_fields(item: Dict[str, Any]) -> List[str]:
    """측정값 8개 중 없거나 숫자가 아닌 필드 이름 (로그용)"""
    return [
        field
        for field in FACE_FEATURE_FIELDS + SKIN_FEATURE_FIELDS
        if _finite_float(item.get(field)) is None
    ]


def extract_features(item: Dict[str, Any]) -> Optional[Tuple[List[float], List[float]]]:
    """
    mediapipe 8개 필드가 모두 (유한한 숫자로) 존재할 때만 (face 6차원, skin 2차원) 반환

    순서는 database/dynamodb_connection.py 와 동일하다. 하나라도 없으면 None -
    호출측은 샘플을 쓰지 않는다 (기본값으로 채우지 않는다).
    """
    face: List[float] = []
    for field in FACE_FEATURE_FIELDS:
        value = _finite_float(item.get(field))
        if value is None:
            return None
        face.append(value)

    skin: List[float] = []
    for field in SKIN_FEATURE_FIELDS:
        value = _finite_float(item.get(field))
        if value is None:
            return None
        skin.append(value)

    return face, skin


def valid_feature_vectors(face_features: Any, skin_features: Any) -> bool:
    """학습 샘플에 쓸 수 있는 (6차원, 2차원) 유한 실수 벡터인지"""
    for values, size in (
        (face_features, len(FACE_FEATURE_FIELDS)),
        (skin_features, len(SKIN_FEATURE_FIELDS)),
    ):
        if values is None or isinstance(values, (str, bytes)):
            return False
        try:
            items: Sequence[Any] = list(values)
        except TypeError:
            return False
        if len(items) != size:
            return False
        if any(_finite_float(v) is None for v in items):
            return False
    return True


def normalize_feature_version(value: Any) -> str:
    """샘플/행의 feature_version 정규화. 없거나 빈 값이면 v1_2d (기존 데이터 전부)"""
    if value is None:
        return FEATURE_VERSION_LEGACY
    text = str(value).strip()
    return text or FEATURE_VERSION_LEGACY


def feature_version_of(item: Dict[str, Any]) -> str:
    """
    분석 행의 face_features 측정 방식 버전

    1. mediapipe_feature_version 이 있으면 그 값
    2. 없지만 mediapipe_head_yaw 가 있으면 v2_3d (3D 측정과 같은 커밋에서 추가된 필드라
       표식 필드가 생기기 전의 3D 행도 구분된다)
    3. 그 외 v1_2d
    """
    explicit = item.get(FEATURE_VERSION_FIELD)
    if explicit is not None and str(explicit).strip():
        return normalize_feature_version(explicit)
    if item.get(_HEAD_POSE_FIELD) is not None:
        return FEATURE_VERSION_3D
    return FEATURE_VERSION_LEGACY


def card_hairstyle_id(card: Any) -> Optional[int]:
    """추천 카드 한 장의 hairstyle_id (트렌드 카드 등 없으면 None)"""
    if not isinstance(card, dict):
        return None
    raw_id = card.get("hairstyle_id")
    if raw_id is None or isinstance(raw_id, bool):
        return None
    try:
        return int(raw_id)
    except (TypeError, ValueError):
        return None


def recommendation_cards(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """분석 항목의 추천 카드 목록 (recommended_styles, 없으면 recommendations)"""
    for field in ("recommended_styles", "recommendations"):
        cards = item.get(field)
        if isinstance(cards, (list, tuple)) and cards:
            return [card for card in cards if isinstance(card, dict)]
    return []


def extract_hairstyle_id(item: Dict[str, Any], style_index: int) -> Optional[int]:
    """recommended_styles[k-1].hairstyle_id (트렌드 스타일은 None -> 스킵)"""
    styles = item.get("recommended_styles") or []
    if not isinstance(styles, (list, tuple)):
        return None
    if style_index > len(styles):
        return None
    return card_hairstyle_id(styles[style_index - 1])
