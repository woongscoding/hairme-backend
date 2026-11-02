"""추천 반응률 보너스 (engagement_v1) - 서빙 쪽

scripts/build_style_engagement.py 가 매일 s3://{MLOPS_S3_BUCKET}/engagement/current.json
에 올리는 "같은 얼굴 그룹(성별 x 얼굴형)의 스타일 반응률" 을 읽어 모델 원점수에
보너스를 더한다.

    보너스 = clip(BETA * (그룹 반응률 - mu) / mu, -CAP, +CAP)   (점수 포인트, 10~95 척도)

그룹 반응률이 없으면 스타일 반응률, 그것도 없으면 mu (보너스 0) 로 내려간다 -
집계가 계층 축소로 만들어 두므로 데이터가 적은 그룹은 자연히 스타일 값에 가깝다.

JSON 은 프로세스 메모리에 캐시하고 TTL(기본 1시간)이 지나면 다시 읽는다. 읽기 실패·
파일 없음·파싱 오류면 None 을 캐시하고(재시도 간격 RETRY_SECONDS) 보너스 없이 기존과
같이 동작한다. 요청 경로에서 매번 S3 를 부르지 않는다.

실험 배정은 탐색 실험(rec_exploration)과 독립된 솔트를 쓴다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from config.settings import settings

logger = logging.getLogger(__name__)

POLICY_ID = "engagement_v1"
ARM_TREATMENT = "treatment"
ARM_CONTROL = "control"

# hairme-analysis 에 남기는 실험 이름 (탐색 실험의 experiment_id / ab_variant 와 별도 필드)
EXPERIMENT_ID = "rec_engagement_v1"

# 탐색 실험(rec-explore-v1)·무료 정책 실험과 독립 배정이 되도록 별도 솔트
ARM_SALT = "rec-engagement-v1"

# 실패를 캐시해 두는 시간 (S3 장애 때 요청마다 재시도하지 않도록)
RETRY_SECONDS = 300.0

SUPPORTED_SCHEMA_VERSIONS = (1,)

UNKNOWN_GENDER = "unknown"
_GENDER_WORDS = {"male": "남성", "female": "여성"}

# 사유 문구 후보 - {who} 는 "둥근형 남성" / 성별을 모르면 "둥근형 얼굴"
REASON_TEMPLATES: Tuple[str, ...] = (
    "나와 같은 {who} 사용자들이 많이 저장한 스타일이에요",
    "{who} 사용자들의 반응이 특히 좋았던 스타일이에요",
    "나와 비슷한 {who} 사용자들이 많이 골라 본 스타일이에요",
)


# ========== 실험 배정 ==========
def assignment_bucket(user_id: Optional[str], image_hash: str) -> float:
    """[0, 1) 버킷. 로그인 사용자는 user_id, 아니면 image_hash 기준"""
    subject = f"user:{user_id}" if user_id else f"image:{image_hash}"
    digest = hashlib.sha256(f"{ARM_SALT}:{subject}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_arm(user_id: Optional[str], image_hash: str, ratio: float) -> str:
    if assignment_bucket(user_id, image_hash) < ratio:
        return ARM_TREATMENT
    return ARM_CONTROL


def normalize_gender(value: Any) -> str:
    if isinstance(value, str) and value.strip().lower() in _GENDER_WORDS:
        return value.strip().lower()
    return UNKNOWN_GENDER


def group_key(gender: Any, face_shape: Any) -> str:
    shape = face_shape.strip() if isinstance(face_shape, str) else ""
    return f"{normalize_gender(gender)}|{shape}"


# ========== 반응률 테이블 ==========
def _finite(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class RateEntry:
    rate: float
    exposures: int
    reactions: float = 0.0  # 반응 가중치 합 (사유 문구 조건용)


def _parse_entries(raw: Any) -> Dict[int, RateEntry]:
    if not isinstance(raw, Mapping):
        raise ValueError("entries must be an object")
    entries: Dict[int, RateEntry] = {}
    for key, value in raw.items():
        hairstyle_id = int(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"entry {key} must be an object")
        rate = _finite(value.get("rate"))
        exposures = _finite(value.get("exposures"))
        if rate is None or rate < 0 or exposures is None or exposures < 0:
            raise ValueError(f"entry {key} has invalid rate/exposures")
        reactions = _finite(value.get("reactions"))
        entries[hairstyle_id] = RateEntry(
            rate, int(exposures), max(reactions or 0.0, 0.0)
        )
    return entries


@dataclass(frozen=True)
class EngagementTable:
    version: str
    mu: float
    styles: Dict[int, RateEntry] = field(default_factory=dict)
    groups: Dict[str, Dict[int, RateEntry]] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> "EngagementTable":
        """JSON -> 테이블. 형식이 틀리면 ValueError"""
        if not isinstance(payload, Mapping):
            raise ValueError("payload must be an object")
        if payload.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported schema_version {payload.get('schema_version')}"
            )
        version = payload.get("version")
        if not isinstance(version, str) or not version:
            raise ValueError("version missing")
        mu = _finite(payload.get("mu"))
        if mu is None or mu < 0:
            raise ValueError("mu missing")
        styles = _parse_entries(payload.get("styles") or {})
        groups_raw = payload.get("groups") or {}
        if not isinstance(groups_raw, Mapping):
            raise ValueError("groups must be an object")
        groups = {str(k): _parse_entries(v) for k, v in groups_raw.items()}
        return cls(version=version, mu=mu, styles=styles, groups=groups)

    def group_rate(self, group: str, hairstyle_id: int) -> Tuple[float, int]:
        """(반응률, 그 단계의 노출 수). 그룹 -> 스타일 -> mu 순으로 내려간다"""
        entry = self.groups.get(group, {}).get(hairstyle_id)
        if entry is not None:
            return entry.rate, entry.exposures
        style = self.styles.get(hairstyle_id)
        if style is not None:
            return style.rate, 0
        return self.mu, 0

    def lift(self, group: str, hairstyle_id: int) -> float:
        """(그룹 반응률 - mu) / mu. mu 가 0 이면 0"""
        if self.mu <= 0:
            return 0.0
        rate, _ = self.group_rate(group, hairstyle_id)
        return (rate - self.mu) / self.mu


def compute_bonus(lift: float, beta: float, cap: float) -> float:
    """clip(beta * lift, -cap, +cap) - 점수 포인트"""
    cap = abs(cap)
    return max(-cap, min(cap, beta * lift))


# ========== S3 캐시 ==========
def _default_loader() -> bytes:
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        region_name=settings.AWS_REGION or "ap-northeast-2",
        config=Config(connect_timeout=2, read_timeout=3, retries={"max_attempts": 1}),
    )
    response = client.get_object(
        Bucket=settings.MLOPS_S3_BUCKET, Key=settings.REC_ENGAGEMENT_S3_KEY
    )
    return response["Body"].read()


class EngagementStore:
    """current.json 을 TTL 동안 메모리에 캐시한다. 실패하면 None (보너스 0)"""

    def __init__(
        self,
        loader: Callable[[], bytes] = _default_loader,
        ttl_seconds: Optional[float] = None,
        retry_seconds: float = RETRY_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._loader = loader
        self._ttl_override = ttl_seconds
        self._retry_seconds = retry_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._table: Optional[EngagementTable] = None
        self._expires_at: Optional[float] = None

    @property
    def ttl_seconds(self) -> float:
        if self._ttl_override is not None:
            return float(self._ttl_override)
        return float(settings.REC_ENGAGEMENT_CACHE_TTL_SECONDS)

    def invalidate(self) -> None:
        with self._lock:
            self._table = None
            self._expires_at = None

    def get(self) -> Optional[EngagementTable]:
        now = self._clock()
        with self._lock:
            if self._expires_at is not None and now < self._expires_at:
                return self._table
            table = self._load()
            ttl = self.ttl_seconds
            delay = ttl if table is not None else min(ttl, self._retry_seconds)
            self._table = table
            self._expires_at = now + max(delay, 0.0)
            return table

    def _load(self) -> Optional[EngagementTable]:
        try:
            body = self._loader()
            table = EngagementTable.from_payload(json.loads(body))
        except Exception as e:  # S3 오류·없음·파싱 오류 모두 보너스 0
            logger.warning(f"[ENGAGEMENT] 반응률 테이블 로드 실패 - 보너스 0: {e}")
            return None
        logger.info(
            f"[ENGAGEMENT] 반응률 테이블 로드: version={table.version} mu={table.mu:.4f} "
            f"styles={len(table.styles)} groups={len(table.groups)}"
        )
        return table


_store: Optional[EngagementStore] = None
_store_lock = threading.Lock()


def get_engagement_store() -> EngagementStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = EngagementStore()
        return _store


def reset_engagement_store(store: Optional[EngagementStore] = None) -> None:
    """테스트용 - 싱글톤 교체"""
    global _store
    with _store_lock:
        _store = store


# ========== 요청 단위 계획 ==========
def _who(gender: str, face_shape: str) -> str:
    word = _GENDER_WORDS.get(gender)
    return f"{face_shape} {word}" if word else f"{face_shape} 얼굴"


def reason_template_index(seed_key: str, hairstyle_id: int) -> int:
    digest = hashlib.sha256(f"{seed_key}:{hairstyle_id}".encode("utf-8")).digest()
    return digest[0] % len(REASON_TEMPLATES)


@dataclass(frozen=True)
class EngagementPlan:
    """한 요청에서 보너스·사유를 계산하는 데 필요한 것"""

    arm: str
    table: EngagementTable
    gender: str
    face_shape: str
    seed_key: str
    beta: float
    cap: float
    reason_min_lift: float
    reason_min_exposures: int
    reason_min_reactions: float = 0.0

    @property
    def group(self) -> str:
        return f"{self.gender}|{self.face_shape}"

    @property
    def treatment(self) -> bool:
        return self.arm == ARM_TREATMENT

    def bonus(self, hairstyle_id: Any) -> float:
        try:
            hid = int(hairstyle_id)
        except (TypeError, ValueError):
            return 0.0
        return compute_bonus(self.table.lift(self.group, hid), self.beta, self.cap)

    def reason(self, hairstyle_id: Any) -> Optional[str]:
        """그룹 반응률이 mu 보다 의미 있게 높고 그룹 노출이 충분할 때만 사유 문구"""
        if not self.face_shape or self.table.mu <= 0:
            return None
        try:
            hid = int(hairstyle_id)
        except (TypeError, ValueError):
            return None
        entry = self.table.groups.get(self.group, {}).get(hid)
        if entry is None or entry.exposures < self.reason_min_exposures:
            return None
        if entry.reactions < self.reason_min_reactions:
            return None
        if (entry.rate - self.table.mu) / self.table.mu < self.reason_min_lift:
            return None
        template = REASON_TEMPLATES[reason_template_index(self.seed_key, hid)]
        return template.format(who=_who(self.gender, self.face_shape))


def build_plan(
    *,
    image_hash: Optional[str],
    user_id: Optional[str],
    gender: Any,
    face_shape: Any,
    store: Optional[EngagementStore] = None,
) -> Optional[EngagementPlan]:
    """플래그 꺼짐 / image_hash 없음 / 테이블 없음이면 None (실험 밖 - 기존과 동일)"""
    if not (settings.REC_ENGAGEMENT_ENABLED and image_hash):
        return None
    table = (store or get_engagement_store()).get()
    if table is None:
        return None
    return EngagementPlan(
        arm=assign_arm(user_id, image_hash, settings.REC_ENGAGEMENT_RATIO),
        table=table,
        gender=normalize_gender(gender),
        face_shape=face_shape.strip() if isinstance(face_shape, str) else "",
        seed_key=image_hash,
        beta=float(settings.REC_ENGAGEMENT_BETA),
        cap=float(settings.REC_ENGAGEMENT_CAP),
        reason_min_lift=float(settings.REC_ENGAGEMENT_REASON_MIN_LIFT),
        reason_min_exposures=int(settings.REC_ENGAGEMENT_REASON_MIN_EXPOSURES),
        reason_min_reactions=float(settings.REC_ENGAGEMENT_REASON_MIN_REACTIONS),
    )
