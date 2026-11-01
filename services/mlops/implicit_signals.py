"""암묵 피드백 (합성·저장·공유) -> 추천 모델 학습 샘플

명시 피드백(좋아요/싫어요)은 분석의 1.8% 만 눌러 재학습이 멈춰 있다. 합성(월 ~500건)과
합성 결과의 저장/공유는 훨씬 자주 일어나므로 이를 학습 샘플로 바꾼다.

샘플 형식은 S3FeedbackStore.save_feedback 과 같은 npz 레이아웃
(face_features / skin_features / style_embedding / ground_truth / metadata) 에
sample_weight(float32[1]) 를 더하고, metadata 에 source="implicit" / signal_type /
rec_rank / ml_rank / propensity 를 넣는다.

    | signal_type      | ground_truth | base_weight |
    | share            | 85           | 1.0         |
    | save             | 80           | 1.0         |
    | synthesize       | 55           | 0.7         |
    | shown_not_chosen | 40           | 0.3         |

    sample_weight = base_weight * clip(1 / propensity, 1, 5)   (propensity 없으면 1.0)

키는 분석 x 스타일 쌍당 1개: feedback/pending/implicit/{analysis_id}_{hairstyle_id}.npz
같은 쌍에 신호가 또 오면 더 강한 신호일 때만 덮어쓴다 (share > save > synthesize >
shown_not_chosen). 기존 객체의 signal_type 은 S3 사용자 메타데이터
(x-amz-meta-signal-type) 에도 넣어 head_object 로 확인한다.

shown_not_chosen 은 한 분석에 처음 synthesize 신호가 들어올 때만, 그 분석의 다른 ML
카드 중 아직 객체가 없는 것에 대해 만든다. "처음" 여부는 pending 밖의 표식 객체
(feedback/implicit_markers/{analysis_id}) 를 조건부 생성해서 판정한다 - 학습 후
pending 이 processed 로 옮겨져도 같은 분석에서 다시 만들어지지 않는다.
트렌드 카드(source != "ml")는 노출 확률이 없어 샘플을 만들지 않는다.

이 모듈은 요청 경로 밖(백그라운드)에서 호출되며 어떤 실패도 밖으로 던지지 않는다.
새 샘플은 feedback/metadata.json 의 pending_count / total_feedback_count 를 올리지만
(덮어쓰기는 올리지 않음) 재학습 트리거는 직접 호출하지 않는다.
"""

import io
import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.logging import log_structured
from services.mlops.feedback_features import (
    card_hairstyle_id,
    extract_features,
    recommendation_cards,
    to_float,
)
from services.mlops.s3_feedback_store import S3_FEEDBACK_PREFIX
from utils.style_preprocessor import normalize_style_name

logger = logging.getLogger(__name__)

try:
    from botocore.exceptions import ClientError, ParamValidationError
except ImportError:  # boto3 미설치 환경 (store 가 비활성화되므로 실제로 쓰이지 않는다)

    class ClientError(Exception):  # type: ignore[no-redef]
        response: Dict[str, Any] = {}

    class ParamValidationError(Exception):  # type: ignore[no-redef]
        pass


SIGNAL_SHARE = "share"
SIGNAL_SAVE = "save"
SIGNAL_SYNTHESIZE = "synthesize"
SIGNAL_SHOWN_NOT_CHOSEN = "shown_not_chosen"

# signal_type -> (ground_truth, base_weight)
SIGNAL_SPECS: Dict[str, Tuple[float, float]] = {
    SIGNAL_SHARE: (85.0, 1.0),
    SIGNAL_SAVE: (80.0, 1.0),
    SIGNAL_SYNTHESIZE: (55.0, 0.7),
    SIGNAL_SHOWN_NOT_CHOSEN: (40.0, 0.3),
}

# 덮어쓰기 우선순위 (클수록 강한 신호)
SIGNAL_PRIORITY: Dict[str, int] = {
    SIGNAL_SHOWN_NOT_CHOSEN: 0,
    SIGNAL_SYNTHESIZE: 1,
    SIGNAL_SAVE: 2,
    SIGNAL_SHARE: 3,
}
# 메타데이터가 없거나 모르는 값인 기존 객체는 가장 약한 것으로 본다
_UNKNOWN_PRIORITY = -1

# 외부(엔드포인트)에서 들어올 수 있는 신호. shown_not_chosen 은 내부에서만 만든다
ACTION_SIGNALS = frozenset({SIGNAL_SHARE, SIGNAL_SAVE, SIGNAL_SYNTHESIZE})

IPS_MIN = 1.0
IPS_MAX = 5.0

IMPLICIT_PREFIX = f"{S3_FEEDBACK_PREFIX}implicit/"
FIRST_SYNTHESIZE_MARKER_PREFIX = "feedback/implicit_markers/"
# x-amz-meta-signal-type (boto3 Metadata 키는 접두사 없이 소문자로 저장된다)
SIGNAL_META_KEY = "signal-type"

SOURCE_IMPLICIT = "implicit"
CARD_SOURCE_ML = "ml"

# analysis_id 는 UUID. S3 키에 들어가므로 안전한 문자만 허용한다
_ANALYSIS_ID_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")

REC_RANK_MIN = 1
REC_RANK_MAX = 5

# put 결과
CREATED = "created"
OVERWRITTEN = "overwritten"
SKIPPED = "skipped"


# ========== 입력 정리 (요청 값은 잘못돼도 무시만 한다) ==========
def parse_analysis_id(value: Any) -> Optional[str]:
    """analysis_id 형식이 맞으면 그대로, 아니면 None"""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if _ANALYSIS_ID_RE.match(text) else None


def parse_optional_int(
    value: Any, minimum: Optional[int] = None, maximum: Optional[int] = None
) -> Optional[int]:
    """정수(또는 정수 문자열)이고 범위 안이면 int, 아니면 None"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    else:
        text = str(value).strip()
        if not re.fullmatch(r"[+-]?\d{1,9}", text):
            return None
        number = int(text)
    if minimum is not None and number < minimum:
        return None
    if maximum is not None and number > maximum:
        return None
    return number


def parse_rec_rank(value: Any) -> Optional[int]:
    """화면에 표시된 카드 순위 1~5"""
    return parse_optional_int(value, REC_RANK_MIN, REC_RANK_MAX)


def parse_hairstyle_id(value: Any) -> Optional[int]:
    return parse_optional_int(value, minimum=0)


# ========== 라벨 / 가중치 ==========
def normalize_propensity(value: Any) -> float:
    """노출 포함확률. 없거나(탐색 꺼짐) 비정상이면 1.0"""
    propensity = to_float(value)
    if propensity is None or not math.isfinite(propensity) or propensity <= 0:
        return 1.0
    return min(propensity, 1.0)


def sample_weight(signal_type: str, propensity: Any) -> float:
    """base_weight x clip(1 / propensity, 1, 5)"""
    _, base_weight = SIGNAL_SPECS[signal_type]
    ips = 1.0 / normalize_propensity(propensity)
    return base_weight * min(max(ips, IPS_MIN), IPS_MAX)


def implicit_key(analysis_id: str, hairstyle_id: int) -> str:
    return f"{IMPLICIT_PREFIX}{analysis_id}_{int(hairstyle_id)}.npz"


def _priority(signal_type: Optional[str]) -> int:
    return SIGNAL_PRIORITY.get(signal_type or "", _UNKNOWN_PRIORITY)


# ========== 카드 매칭 ==========
def find_card(
    cards: List[Dict[str, Any]],
    hairstyle_id: Optional[int] = None,
    style_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """hairstyle_id 로 먼저, 없으면 style_name(띄어쓰기 무시)으로 카드를 찾는다"""
    if hairstyle_id is not None:
        for card in cards:
            if card_hairstyle_id(card) == hairstyle_id:
                return card

    normalized = normalize_style_name(style_name or "")
    if normalized:
        for card in cards:
            if normalize_style_name(str(card.get("style_name") or "")) == normalized:
                return card
    return None


def _is_ml_card(card: Dict[str, Any]) -> bool:
    return card.get("source") == CARD_SOURCE_ML and card_hairstyle_id(card) is not None


# ========== npz ==========
def build_sample_npz(
    *,
    analysis: Dict[str, Any],
    analysis_id: str,
    card: Dict[str, Any],
    hairstyle_id: int,
    signal_type: str,
    face_features: List[float],
    skin_features: List[float],
    style_embedding: np.ndarray,
    rec_rank: Optional[int],
    timestamp: datetime,
) -> bytes:
    """save_feedback 과 같은 레이아웃 + sample_weight + 암묵 메타"""
    ground_truth, _ = SIGNAL_SPECS[signal_type]
    propensity = normalize_propensity(card.get("propensity"))
    weight = sample_weight(signal_type, propensity)
    ml_rank = parse_optional_int(_int_like(card.get("ml_rank")), minimum=1)

    metadata = {
        "analysis_id": analysis_id,
        "face_shape": analysis.get("face_shape") or "",
        "skin_tone": analysis.get("personal_color") or "",
        "hairstyle_id": hairstyle_id,
        "style_name": card.get("style_name"),
        "timestamp": timestamp.isoformat(),
        "source": SOURCE_IMPLICIT,
        "signal_type": signal_type,
        "rec_rank": rec_rank,
        "ml_rank": ml_rank,
        "propensity": propensity,
        "sample_weight": weight,
        "model_version": analysis.get("model_version"),
    }

    npz_data = {
        "face_features": np.array(face_features, dtype=np.float32),
        "skin_features": np.array(skin_features, dtype=np.float32),
        "style_embedding": np.asarray(style_embedding).astype(np.float32),
        "ground_truth": np.array([ground_truth], dtype=np.float32),
        "sample_weight": np.array([weight], dtype=np.float32),
        "metadata": np.array([json.dumps(metadata, ensure_ascii=False)], dtype=str),
    }

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **npz_data)
    return buffer.getvalue()


def _int_like(value: Any) -> Any:
    """DynamoDB 에서 float 로 돌아온 정수(3.0)를 int 로"""
    number = to_float(value)
    if number is None or not math.isfinite(number) or number != int(number):
        return None
    return int(number)


# ========== 기록기 ==========
def _default_analysis_getter(analysis_id: str) -> Optional[Dict[str, Any]]:
    from database.dynamodb_connection import get_analysis

    return get_analysis(analysis_id)


def _default_store():
    from services.mlops.s3_feedback_store import get_s3_feedback_store

    return get_s3_feedback_store()


def _mlops_enabled() -> bool:
    # S3FeedbackStore / training_trigger 와 같은 판단 (환경변수)
    return os.getenv("MLOPS_ENABLED", "false").lower() == "true"


def _error_code(error: Exception) -> str:
    response = getattr(error, "response", None) or {}
    return str((response.get("Error") or {}).get("Code", ""))


class ImplicitSignalRecorder:
    """S3FeedbackStore 의 S3 클라이언트·임베딩·metadata.json 을 그대로 쓴다"""

    def __init__(self, store, analysis_getter: Callable[[str], Optional[Dict]]):
        self.store = store
        self.s3 = store.s3_client
        self.bucket = store.bucket_name
        self.get_analysis = analysis_getter
        # IfNoneMatch 를 못 쓰는 환경이면 head 후 put 으로 내려간다
        self._conditional_put = True

    # ----- S3 저수준 -----
    def _head_signal(self, key: str) -> Tuple[bool, Optional[str]]:
        """(존재 여부, 기존 signal_type)"""
        try:
            response = self.s3.head_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            if _error_code(e) in ("404", "NoSuchKey", "NotFound"):
                return False, None
            raise
        return True, (response.get("Metadata") or {}).get(SIGNAL_META_KEY)

    def _put(self, key: str, body: bytes, signal_type: Optional[str], **extra) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/octet-stream",
            Metadata={SIGNAL_META_KEY: signal_type} if signal_type else {},
            **extra,
        )

    def _put_if_absent(self, key: str, body: bytes, signal_type: Optional[str]) -> bool:
        """객체가 없을 때만 만든다. 만들었으면 True"""
        if self._conditional_put:
            try:
                self._put(key, body, signal_type, IfNoneMatch="*")
                return True
            except ClientError as e:
                code = _error_code(e)
                if code in ("PreconditionFailed", "ConditionalRequestConflict", "412"):
                    return False
                if code not in ("NotImplemented", "InvalidArgument", "501"):
                    raise
                logger.info("조건부 put 미지원 - head 후 put 으로 전환")
                self._conditional_put = False
            except ParamValidationError:
                logger.info("조건부 put 미지원(boto3) - head 후 put 으로 전환")
                self._conditional_put = False

        exists, _ = self._head_signal(key)
        if exists:
            return False
        self._put(key, body, signal_type)
        return True

    def _write_sample(self, key: str, body: bytes, signal_type: str) -> str:
        """더 강한 신호일 때만 덮어쓴다. created / overwritten / skipped"""
        if signal_type == SIGNAL_SHOWN_NOT_CHOSEN:
            # 가장 약한 신호라 기존 객체가 있으면 무조건 둔다 - head 없이 조건부 생성
            return CREATED if self._put_if_absent(key, body, signal_type) else SKIPPED

        exists, existing = self._head_signal(key)
        if not exists:
            if self._put_if_absent(key, body, signal_type):
                return CREATED
            # 경합: 그 사이 다른 요청이 같은 쌍을 만들었다
            _, existing = self._head_signal(key)

        if _priority(signal_type) <= _priority(existing):
            return SKIPPED
        self._put(key, body, signal_type)
        return OVERWRITTEN

    def _claim_first_synthesize(self, analysis_id: str) -> bool:
        """이 분석의 첫 synthesize 신호면 True (표식 객체를 조건부 생성)"""
        return self._put_if_absent(
            f"{FIRST_SYNTHESIZE_MARKER_PREFIX}{analysis_id}", b"", None
        )

    def _bump_metadata(self, created: int) -> None:
        """save_feedback 과 같은 방식으로 카운터 갱신 (재학습 트리거는 하지 않는다)"""
        metadata = self.store._get_metadata()
        metadata["total_feedback_count"] = (
            metadata.get("total_feedback_count", 0) + created
        )
        metadata["pending_count"] = metadata.get("pending_count", 0) + created
        metadata["last_feedback_at"] = datetime.now(timezone.utc).isoformat()
        self.store._save_metadata(metadata)

    # ----- 기록 -----
    def record(
        self,
        analysis_id: str,
        signal_type: str,
        hairstyle_id: Optional[int] = None,
        style_name: Optional[str] = None,
        rec_rank: Optional[int] = None,
    ) -> Dict[str, Any]:
        analysis = self.get_analysis(analysis_id)
        if not analysis:
            return {"recorded": False, "reason": "analysis_not_found"}

        features = extract_features(analysis)
        if features is None:
            return {"recorded": False, "reason": "no_features"}
        face_features, skin_features = features

        cards = recommendation_cards(analysis)
        card = find_card(cards, hairstyle_id, style_name)
        if card is None:
            return {"recorded": False, "reason": "card_not_found"}
        if not _is_ml_card(card):
            return {"recorded": False, "reason": "not_ml_card"}

        chosen_id = card_hairstyle_id(card)
        timestamp = datetime.now(timezone.utc)
        created = 0

        def _write(target: Dict[str, Any], target_id: int, signal: str, rank) -> str:
            embedding = self.store.get_style_embedding(target_id)
            if embedding is None:
                return SKIPPED
            body = build_sample_npz(
                analysis=analysis,
                analysis_id=analysis_id,
                card=target,
                hairstyle_id=target_id,
                signal_type=signal,
                face_features=face_features,
                skin_features=skin_features,
                style_embedding=embedding,
                rec_rank=rank,
                timestamp=timestamp,
            )
            return self._write_sample(implicit_key(analysis_id, target_id), body, signal)

        chosen_rank = rec_rank if rec_rank is not None else _card_rank(card)
        result = _write(card, chosen_id, signal_type, chosen_rank)
        if result == CREATED:
            created += 1

        not_chosen = 0
        if signal_type == SIGNAL_SYNTHESIZE and self._claim_first_synthesize(
            analysis_id
        ):
            for other in cards:
                other_id = card_hairstyle_id(other)
                if not _is_ml_card(other) or other_id == chosen_id:
                    continue
                if (
                    _write(other, other_id, SIGNAL_SHOWN_NOT_CHOSEN, _card_rank(other))
                    == CREATED
                ):
                    created += 1
                    not_chosen += 1

        if created:
            self._bump_metadata(created)

        return {
            "recorded": result in (CREATED, OVERWRITTEN),
            "reason": result,
            "hairstyle_id": chosen_id,
            "shown_not_chosen": not_chosen,
            "created": created,
        }


def _card_rank(card: Dict[str, Any]) -> Optional[int]:
    return parse_optional_int(_int_like(card.get("rank")), minimum=1)


def record_style_signal(
    analysis_id: Any,
    signal_type: str,
    *,
    hairstyle_id: Any = None,
    style_name: Optional[str] = None,
    rec_rank: Any = None,
    store=None,
    analysis_getter: Optional[Callable[[str], Optional[Dict]]] = None,
) -> Dict[str, Any]:
    """신호 하나를 학습 샘플로 기록 (best effort - 예외를 던지지 않는다)

    요청 경로를 막지 않도록 BackgroundTasks 에서 호출한다.
    """
    parsed_id = parse_analysis_id(analysis_id)
    summary: Dict[str, Any] = {"recorded": False, "reason": None}
    try:
        if signal_type not in ACTION_SIGNALS:
            summary["reason"] = "invalid_signal"
        elif parsed_id is None:
            summary["reason"] = "invalid_analysis_id"
        elif not _mlops_enabled():
            summary["reason"] = "mlops_disabled"
        else:
            store = store if store is not None else _default_store()
            if not getattr(store, "enabled", False) or store.s3_client is None:
                summary["reason"] = "store_disabled"
            else:
                recorder = ImplicitSignalRecorder(
                    store, analysis_getter or _default_analysis_getter
                )
                summary = recorder.record(
                    parsed_id,
                    signal_type,
                    hairstyle_id=parse_hairstyle_id(hairstyle_id),
                    style_name=style_name,
                    rec_rank=parse_rec_rank(rec_rank),
                )
    except Exception as e:
        logger.warning(f"암묵 피드백 기록 실패 (무시): {e}")
        summary = {"recorded": False, "reason": "error"}

    log_structured(
        "implicit_signal",
        {
            "analysis_id": parsed_id,
            "signal_type": signal_type,
            "rec_rank": parse_rec_rank(rec_rank),
            **summary,
        },
    )
    return summary
