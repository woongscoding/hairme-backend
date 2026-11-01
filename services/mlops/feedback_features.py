"""hairme-analysis 항목에서 학습 샘플 입력을 꺼내는 공통 함수

scripts/backfill_feedback_to_s3.py (명시 피드백 백필)와
services/mlops/implicit_signals.py (암묵 피드백) 가 같은 규칙을 쓰도록 모아 둔다.
측정값 순서는 database/dynamodb_connection.py 의 저장 순서와 같다.
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

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
