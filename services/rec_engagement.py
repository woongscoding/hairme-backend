"""추천 반응률 보너스 (engagement_v1) - 서빙 쪽

scripts/build_style_engagement.py 가 매일 s3://{MLOPS_S3_BUCKET}/engagement/current.json
에 올리는 "같은 얼굴 그룹(성별 x 얼굴형)의 스타일 반응률" 을 읽어 모델 원점수에
보너스를 더한다.

    lift   = (셀 반응률 - 기준선) / 기준선   (기준선 = 이 스타일을 뺀 그룹 평균 반응률)
    보너스 = clip(BETA * lift, -CAP, +CAP)   (점수 포인트, 10~95 척도)

그룹 기저율 기준이라 반응이 원래 많은 그룹이어도 모든 카드에 일괄 + 가 붙지 않는다.
셀이 없으면 스타일 반응률 / mu, 그것도 없으면 0 으로 내려간다 - 집계가 계층 축소로
만들어 두므로 데이터가 적은 셀은 자연히 "그룹 평균 x 스타일 상대값" 에 가깝다.

JSON 은 프로세스 메모리에 캐시하고 TTL(기본 1시간)이 지나면 다시 읽는다.
    - 읽기 실패 / 파일 없음 / 파싱 오류 / 검증 실패(mu <= 0, 빈 styles, generated_at 이
      REC_ENGAGEMENT_MAX_AGE_DAYS 초과) 는 구조화 로그 engagement_table_rejected 를 남긴다.
    - 갱신에 실패해도 마지막 정상 테이블이 아직 신선하면 그대로 쓴다 (stale-while-error).
      없거나 신선도 한도를 넘으면 None - 보너스 없이 기존과 같이 동작한다.
    - 실패 후 재시도는 RETRY_SECONDS 뒤. 요청 경로에서 매번 S3 를 부르지 않는다.

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
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from config.settings import settings
from core.logging import log_structured

logger = logging.getLogger(__name__)

POLICY_ID = "engagement_v1"
ARM_TREATMENT = "treatment"
ARM_CONTROL = "control"

# hairme-analysis 에 남기는 실험 이름 (탐색 실험의 experiment_id / ab_variant 와 별도 필드)
EXPERIMENT_ID = "rec_engagement_v1"

# 탐색 실험(rec-explore-v1)·무료 정책 실험과 독립 배정이 되도록 별도 솔트
ARM_SALT = "rec-engagement-v1"

# 실패 후 다시 읽기까지 (S3 장애 때 요청마다 재시도하지 않도록)
RETRY_SECONDS = 300.0

SUPPORTED_SCHEMA_VERSIONS = (1,)

UNKNOWN_GENDER = "unknown"
_GENDER_WORDS = {"male": "남성", "female": "여성"}

# 사유 문구 후보 - {who} 는 "둥근형 남성" / 성별을 모르면 "둥근형 얼굴".
# 근거는 save/share 한 고유 주체 수 (합성·좋아요만으로는 나가지 않는다).
REASON_TEMPLATES: Tuple[str, ...] = (
    "나와 같은 {who} 사용자들이 많이 저장한 스타일이에요",
    "{who} 사용자들의 반응이 특히 좋았던 스타일이에요",
    "나와 비슷한 {who} 사용자들이 많이 저장·공유한 스타일이에요",
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
class TableRejected(ValueError):
    """형식은 읽혔지만 쓰면 안 되는 테이블 (reason 은 로그용 짧은 코드)"""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


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
    reactions: float = 0.0  # 반응 가중치 합
    subjects: int = 0  # save/share 한 고유 주체 수 (사유 문구 조건)
    base: Optional[float] = None  # lift 기준선 (이 스타일을 뺀 그룹 평균)


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
        reactions = _finite(value.get("reactions")) or 0.0
        subjects = _finite(value.get("subjects")) or 0.0
        base = _finite(value.get("base"))
        entries[hairstyle_id] = RateEntry(
            rate,
            int(exposures),
            max(reactions, 0.0),
            max(int(subjects), 0),
            base if base is not None and base > 0 else None,
        )
    return entries


def _parse_group_means(raw: Any) -> Dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("group_totals must be an object")
    means: Dict[str, float] = {}
    for key, value in raw.items():
        mean = _finite(value.get("mean_rate")) if isinstance(value, Mapping) else None
        if mean is not None and mean > 0:
            means[str(key)] = mean
    return means


@dataclass(frozen=True)
class EngagementTable:
    version: str
    mu: float
    styles: Dict[int, RateEntry] = field(default_factory=dict)
    groups: Dict[str, Dict[int, RateEntry]] = field(default_factory=dict)
    group_means: Dict[str, float] = field(default_factory=dict)
    generated_at: Optional[datetime] = None

    @classmethod
    def from_payload(cls, payload: Any) -> "EngagementTable":
        """JSON -> 테이블. 형식이 틀리면 ValueError, 쓰면 안 되면 TableRejected"""
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
        if mu is None:
            raise ValueError("mu missing")
        if mu <= 0:
            raise TableRejected("mu_not_positive", str(mu))
        styles = _parse_entries(payload.get("styles") or {})
        if not styles:
            raise TableRejected("empty_styles")
        groups_raw = payload.get("groups") or {}
        if not isinstance(groups_raw, Mapping):
            raise ValueError("groups must be an object")
        groups = {str(k): _parse_entries(v) for k, v in groups_raw.items()}
        generated_at = _parse_time(payload.get("generated_at"))
        return cls(
            version=version,
            mu=mu,
            styles=styles,
            groups=groups,
            group_means=_parse_group_means(payload.get("group_totals")),
            generated_at=generated_at,
        )

    def is_fresh(self, now: datetime, max_age: timedelta) -> bool:
        """generated_at 이 max_age 안이면 True (generated_at 없으면 신선하지 않음)"""
        if self.generated_at is None:
            return False
        return now - self.generated_at <= max_age

    def group_mean(self, group: str) -> float:
        return self.group_means.get(group, self.mu)

    def lift(self, group: str, hairstyle_id: int) -> float:
        """셀이 있으면 (셀 - 기준선) / 기준선, 없으면 (스타일 - mu) / mu, 없으면 0

        기준선은 셀의 base(이 스타일을 뺀 그룹 평균), 없으면 그룹 평균, 없으면 mu.
        """
        entry = self.groups.get(group, {}).get(hairstyle_id)
        if entry is not None:
            base = entry.base if entry.base is not None else self.group_mean(group)
            return (entry.rate - base) / base if base > 0 else 0.0
        style = self.styles.get(hairstyle_id)
        if style is not None and self.mu > 0:
            return (style.rate - self.mu) / self.mu
        return 0.0


def _parse_time(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


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


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _reject(reason: str, detail: str = "", version: Optional[str] = None) -> None:
    logger.warning(f"[ENGAGEMENT] 반응률 테이블 거부 ({reason}) - 보너스 0: {detail}")
    log_structured(
        "engagement_table_rejected",
        {"reason": reason, "detail": detail[:200], "version": version},
    )


class EngagementStore:
    """current.json 을 TTL 동안 메모리에 캐시한다 (stale-while-error + 신선도 한도)"""

    def __init__(
        self,
        loader: Callable[[], bytes] = _default_loader,
        ttl_seconds: Optional[float] = None,
        retry_seconds: float = RETRY_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        max_age_days: Optional[float] = None,
        wall_clock: Callable[[], datetime] = _utcnow,
    ):
        self._loader = loader
        self._ttl_override = ttl_seconds
        self._retry_seconds = retry_seconds
        self._clock = clock
        self._max_age_override = max_age_days
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._table: Optional[EngagementTable] = None
        self._expires_at: Optional[float] = None

    @property
    def ttl_seconds(self) -> float:
        if self._ttl_override is not None:
            return float(self._ttl_override)
        return float(settings.REC_ENGAGEMENT_CACHE_TTL_SECONDS)

    @property
    def max_age(self) -> timedelta:
        days = self._max_age_override
        if days is None:
            days = float(settings.REC_ENGAGEMENT_MAX_AGE_DAYS)
        return timedelta(days=days)

    def invalidate(self) -> None:
        with self._lock:
            self._table = None
            self._expires_at = None

    def _fresh(self, table: Optional[EngagementTable]) -> bool:
        return table is not None and table.is_fresh(self._wall_clock(), self.max_age)

    def get(self) -> Optional[EngagementTable]:
        now = self._clock()
        with self._lock:
            if self._expires_at is not None and now < self._expires_at:
                if self._table is not None and not self._fresh(self._table):
                    # 캐시된 사이 신선도 한도를 넘었다 (집계가 며칠째 안 돈 경우)
                    _reject("stale", "cached table expired", self._table.version)
                    self._table = None
                return self._table

            loaded = self._load()
            if loaded is not None:
                self._table = loaded
                self._expires_at = now + max(self.ttl_seconds, 0.0)
                return loaded

            # 실패: 마지막 정상 테이블이 아직 신선하면 계속 쓴다
            if self._table is not None and not self._fresh(self._table):
                _reject("stale", "last good table too old", self._table.version)
                self._table = None
            self._expires_at = now + max(
                min(self.ttl_seconds, self._retry_seconds), 0.0
            )
            return self._table

    def _load(self) -> Optional[EngagementTable]:
        try:
            body = self._loader()
        except Exception as e:  # S3 오류·없음
            _reject("load_error", str(e))
            return None
        try:
            table = EngagementTable.from_payload(json.loads(body))
        except TableRejected as e:
            _reject(e.reason, str(e))
            return None
        except Exception as e:  # 파싱·형식 오류
            _reject("parse_error", str(e))
            return None
        if not self._fresh(table):
            _reject(
                "stale",
                f"generated_at={table.generated_at.isoformat() if table.generated_at else None}",
                table.version,
            )
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
    reason_min_subjects: int = 0

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
        """셀 lift 가 의미 있게 높고 노출·save/share 고유 주체가 충분할 때만 사유 문구"""
        if not self.face_shape:
            return None
        try:
            hid = int(hairstyle_id)
        except (TypeError, ValueError):
            return None
        entry = self.table.groups.get(self.group, {}).get(hid)
        if entry is None or entry.exposures < self.reason_min_exposures:
            return None
        if entry.subjects < self.reason_min_subjects:
            return None
        if self.table.lift(self.group, hid) < self.reason_min_lift:
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
        reason_min_subjects=int(settings.REC_ENGAGEMENT_REASON_MIN_SUBJECTS),
    )
