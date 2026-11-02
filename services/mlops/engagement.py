"""스타일 반응률 집계 (추천 보너스용)

추천 모델 점수에 "같은 얼굴 그룹(성별 x 얼굴형)에서 이 스타일이 화면에 나갔을 때
얼마나 반응을 얻었는가" 를 보너스로 더하기 위한 통계를 만든다.
서빙 쪽은 services/rec_engagement.py, 정기 실행은 scripts/build_style_engagement.py.

노출 (exposure)
    hairme-analysis 의 분석별 카드 중 화면에 나간 ML 카드(source == "ml").
    트렌드 카드는 노출 확률·hairstyle_id 가 없어 개수만 따로 센다.
    사유 문구가 반응률 문구로 바뀐 카드(engagement_reason == true)는 "많이 저장한
    스타일이에요" 라는 말 자체가 반응을 늘리는 사회적 증거 루프가 되므로 집계에서
    빼고 개수만 센다.

반응 (reaction)
    S3 hairme-mlops 의 학습 샘플 npz (feedback/pending/** 와 트레이너가 옮긴
    feedback/processed/**). 암묵 신호는 metadata.source == "implicit" 의
    signal_type, 명시 피드백은 metadata.feedback (good/bad).
    analysis_id x hairstyle_id 쌍마다 가중치가 가장 큰 신호 1개만 쓴다.
    노출이 없는 쌍(창 밖 분석, 트렌드 카드, 노출 기록 이전 데이터)은 버리고 개수만 보고.
    save/share 는 서버가 그 쌍을 실제로 합성했고 result_id 가 맞을 때만 npz 가
    생긴다 (services/mlops/implicit_signals.py).

주체당 기여 상한
    /api/v2/analyze-hybrid 는 비인증이고 같은 사진을 다시 올리면 새 analysis_id 를
    준다. 한 사람이 같은 사진으로 분석·합성·공유를 반복해 셀을 움직이지 못하도록
    주체(로그인이면 user_id, 아니면 image_hash) x 그룹 x 스타일마다 노출 1, 반응은
    그 주체의 최댓값 1개로 센다. 주체 식별자는 메모리에서만 해시로 쓰고 출력하지 않는다.

위치 보정 (기본 꺼짐)
    1위 카드가 3위보다 반응을 더 받는 위치 효과를 f_rank 로 나눠 센다. 단순히
    "순위별 반응률" 로 추정하면 늘 1위에 나오는 좋은 스타일의 품질이 위치 효과로
    잘못 잡혀 과보정된다. 그래서 켜면 둘 이상의 순위에 노출된 스타일만으로
    스타일 고정효과(rho_s) x 위치(f_r) 를 번갈아 맞춰 추정한다. 탐색이 꺼진 지금은
    같은 스타일이 여러 순위에 나오는 경우가 적어 추정이 불안정하므로 기본은 꺼짐.

계층적 축소 (반응률 = 반응 가중합 / 노출)
    mu        = 전체 반응합 / 전체 노출
    style     = (R_s + m1 * mu) / (N_s + m1)
    group     = (R_g + ag * 1) / (N_g + ag / mu)          그룹 평균 반응률 (mu 로 축소)
    cell      = (R_gs + m2 * style * group / mu) / (N_gs + m2)
    base_gs   = 이 스타일을 뺀 그룹 평균 (leave-one-out, mu 로 축소) - 서빙 lift 기준선
    m1 = a1 / mu, m2 = a2 / mu  (a = "반응 가중치 단위" 사전 강도)

    셀의 사전값을 style * (group / mu) 로 두면 반응이 많은 그룹이라도 셀 데이터가
    적은 스타일은 "그룹 평균 x 스타일 상대값" 으로 수렴한다. 서빙은 lift 를
    (셀 - 그룹 평균) / 그룹 평균 으로 계산하므로, 그룹 전체가 반응이 높아도 모든
    카드에 일괄 + 보너스가 붙지 않는다. 기준선은 셀마다 그 스타일을 뺀 그룹 평균
    (base) 이라, 그룹 노출의 대부분을 차지하는 스타일도 자기 반응에 기준선이 끌려
    올라가지 않는다.

    m 을 노출 수로 고정하면 반응이 드문 시기(예: 명시 피드백만 있을 때 mu ~ 0.01)에
    좋아요 1~2개로 반응률이 몇 배가 되어 보너스가 상한에 붙는다. a 를 반응 단위로
    두면 사전 분포가 mu 에 비례해 커져서 mu 의 크기와 무관하게 같은 정도로 축소된다.
    (감마-포아송 모형의 사후평균과 같은 형태). 기본값 근거는 scripts/build_style_engagement.py.

얼굴형 신뢰 구간
    2026-09-17 얼굴형 재보정 이전 행은 98% 가 계란형이라 그룹 통계에 넣지 않는다.
    이 행들의 노출·반응은 전역 mu 와 스타일 반응률에만 들어간다 (face_shape_since).

성별
    hairme-analysis 에 gender 가 저장되기 시작한 것은 2026-09-20 11:33 UTC 부터다.
    그 이전 행(또는 male/female 이 아닌 값)은 "unknown" 그룹으로 센다.

출력 JSON 에는 hairstyle_id·그룹 키·숫자만 넣는다 (user_id / device_id / analysis_id /
image_hash 없음). subjects 는 save/share 한 고유 주체 "수" 다.
auth_subjects 는 그중 로그인 사용자 수다. 비로그인 주체(image_hash)는 사진만 바꾸면
새 주체가 되므로, "많이 저장한" 사유 문구 조건은 auth_subjects 로만 판단한다.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from services.mlops.feedback_features import card_hairstyle_id, recommendation_cards

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# ========== 신호 종류 / 기본 가중치 ==========
KIND_SHARE = "share"
KIND_SAVE = "save"
KIND_SYNTHESIZE = "synthesize"
KIND_SHOWN_NOT_CHOSEN = "shown_not_chosen"
KIND_EXPLICIT_GOOD = "explicit_good"
KIND_EXPLICIT_BAD = "explicit_bad"

IMPLICIT_KINDS = (KIND_SHARE, KIND_SAVE, KIND_SYNTHESIZE, KIND_SHOWN_NOT_CHOSEN)
EXPLICIT_KINDS = (KIND_EXPLICIT_GOOD, KIND_EXPLICIT_BAD)
# "많이 저장한" 사유 문구의 근거가 되는 신호 (합성·좋아요는 세지 않는다)
SAVE_LIKE_KINDS = (KIND_SAVE, KIND_SHARE)

DEFAULT_WEIGHTS: Dict[str, float] = {
    KIND_SYNTHESIZE: 1.0,
    KIND_SAVE: 3.0,
    # 앱은 공유 시트를 "연" 시점(취소 포함)에 share 를 보낸다. 실제 공유가 확인되지
    # 않으므로 save 보다 강한 신호로 보지 않는다 (--weights share=4 로 바꿀 수 있다).
    KIND_SHARE: 3.0,
    KIND_SHOWN_NOT_CHOSEN: 0.0,
    KIND_EXPLICIT_GOOD: 2.0,
    KIND_EXPLICIT_BAD: 0.0,
}

# 같은 가중치일 때 어느 신호로 기록할지 (보고·subjects 용 - 반응합에는 영향 없음)
_KIND_TIEBREAK = {
    KIND_SHOWN_NOT_CHOSEN: 0,
    KIND_EXPLICIT_BAD: 1,
    KIND_SYNTHESIZE: 2,
    KIND_EXPLICIT_GOOD: 3,
    KIND_SAVE: 4,
    KIND_SHARE: 5,
}

_POSITIVE_FEEDBACK = {"good", "like"}
_NEGATIVE_FEEDBACK = {"bad", "dislike"}

# ========== 기본 파라미터 (근거: scripts/build_style_engagement.py 머리말) ==========
DEFAULT_WINDOW_DAYS = 60
DEFAULT_A1 = 20.0  # 스타일 반응률 사전 강도 (반응 가중치 단위)
DEFAULT_A2 = 20.0  # 셀(그룹 x 스타일) 반응률 사전 강도
DEFAULT_AG = 20.0  # 그룹 평균 반응률 사전 강도
DEFAULT_POSITION_ADJUST = False
DEFAULT_POSITION_PRIOR = 20.0  # 위치 보정 f_rank 사전 강도 (반응 가중치 단위)
FACE_SHAPE_RELIABLE_SINCE = "2026-09-17"

UNKNOWN_GENDER = "unknown"
KNOWN_GENDERS = ("male", "female")
ANONYMOUS_USER = "anonymous"

CARD_SOURCE_ML = "ml"

# 주체 해시 솔트 (메모리 안에서만 쓴다 - 출력에 남지 않는다)
_SUBJECT_SALT = b"hairme-engagement-subject"
# 주체 종류 접두사 (로그인 사용자 / 비로그인 사진). 해시 앞에 붙어 메모리에서만 쓰인다
AUTH_SUBJECT_PREFIX = "u:"
IMAGE_SUBJECT_PREFIX = "i:"


# ========== 값 정리 ==========
def normalize_gender(value: Any) -> str:
    """male / female 이외(None, neutral 등)는 unknown"""
    if isinstance(value, str) and value.strip().lower() in KNOWN_GENDERS:
        return value.strip().lower()
    return UNKNOWN_GENDER


def group_key(gender: Any, face_shape: Any) -> Optional[str]:
    """그룹 키 "male|각진형". 얼굴형이 없으면 None (그룹 통계에서 제외)"""
    if not isinstance(face_shape, str) or not face_shape.strip():
        return None
    return f"{normalize_gender(gender)}|{face_shape.strip()}"


def subject_of(item: Mapping[str, Any]) -> Optional[str]:
    """기여 상한 단위. 로그인 user_id 우선, 없으면 image_hash. 해시만 돌려준다"""
    user_id = item.get("user_id")
    if isinstance(user_id, str) and user_id and user_id != ANONYMOUS_USER:
        prefix, raw = AUTH_SUBJECT_PREFIX, f"user:{user_id}"
    else:
        image_hash = item.get("image_hash")
        if not isinstance(image_hash, str) or not image_hash:
            return None
        prefix, raw = IMAGE_SUBJECT_PREFIX, f"image:{image_hash}"
    digest = hashlib.sha256(_SUBJECT_SALT + raw.encode("utf-8")).hexdigest()[:20]
    return prefix + digest


def is_auth_subject(subject: Optional[str]) -> bool:
    """로그인 사용자 주체인지 (subject_of 가 만든 접두사로 판단)"""
    return isinstance(subject, str) and subject.startswith(AUTH_SUBJECT_PREFIX)


def parse_timestamp(value: Any) -> Optional[datetime]:
    """created_at(ISO8601) -> aware datetime. 실패 시 None"""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _card_rank(card: Mapping[str, Any]) -> Optional[int]:
    raw = card.get("rank")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number != int(number) or number < 1:
        return None
    return int(number)


# ========== 노출 ==========
@dataclass(frozen=True)
class Exposure:
    """화면에 나간 ML 카드 1장

    group 이 None 이면 그룹 통계에서 제외 (얼굴형 재보정 이전 / 얼굴형 없음).
    subject 가 None 이면 analysis_id 를 주체로 본다.
    social_proof 는 반응률 사유 문구가 붙은 카드 (집계 제외).
    """

    analysis_id: str
    hairstyle_id: int
    rank: Optional[int]
    group: Optional[str]
    subject: Optional[str] = None
    social_proof: bool = False


@dataclass
class ExposureStats:
    analyses: int = 0
    ml_exposures: int = 0
    trending_exposures: int = 0
    social_proof_exposures: int = 0
    skipped_out_of_window: int = 0
    skipped_no_cards: int = 0
    gender_unknown_analyses: int = 0
    face_shape_untrusted_analyses: int = 0
    no_subject_analyses: int = 0
    first_gender_at: Optional[str] = None
    trending_by_name: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        top_trending = sorted(
            self.trending_by_name.items(), key=lambda kv: kv[1], reverse=True
        )[:10]
        return {
            "analyses": self.analyses,
            "ml_exposures": self.ml_exposures,
            "trending_exposures": self.trending_exposures,
            "social_proof_exposures": self.social_proof_exposures,
            "skipped_out_of_window": self.skipped_out_of_window,
            "skipped_no_cards": self.skipped_no_cards,
            "gender_unknown_analyses": self.gender_unknown_analyses,
            "face_shape_untrusted_analyses": self.face_shape_untrusted_analyses,
            "no_subject_analyses": self.no_subject_analyses,
            "first_gender_at": self.first_gender_at,
            "trending_top": [
                {"style_name": k, "exposures": v} for k, v in top_trending
            ],
        }


def collect_exposures(
    items: Iterable[Mapping[str, Any]],
    since: datetime,
    until: datetime,
    face_shape_since: Optional[datetime] = None,
) -> Tuple[List[Exposure], ExposureStats]:
    """hairme-analysis 항목 -> ML 카드 노출 목록

    since <= created_at < until 인 분석만. 같은 분석 안에서 같은 hairstyle_id 는 1번만.
    face_shape_since 이전 분석은 group=None (전역·스타일 통계에만 들어간다).
    """
    exposures: List[Exposure] = []
    stats = ExposureStats()
    first_gender: Optional[datetime] = None

    for item in items:
        created = parse_timestamp(item.get("created_at"))
        analysis_id = item.get("analysis_id")
        if created is None or not analysis_id or not (since <= created < until):
            stats.skipped_out_of_window += 1
            continue
        cards = recommendation_cards(dict(item))
        if not cards:
            stats.skipped_no_cards += 1
            continue

        stats.analyses += 1
        gender = normalize_gender(item.get("gender"))
        if gender == UNKNOWN_GENDER:
            stats.gender_unknown_analyses += 1
        elif first_gender is None or created < first_gender:
            first_gender = created

        trusted = face_shape_since is None or created >= face_shape_since
        if not trusted:
            stats.face_shape_untrusted_analyses += 1
        group = group_key(gender, item.get("face_shape")) if trusted else None
        subject = subject_of(item)
        if subject is None:
            stats.no_subject_analyses += 1

        seen = set()
        for card in cards:
            if card.get("source") != CARD_SOURCE_ML:
                stats.trending_exposures += 1
                name = str(card.get("style_name") or "").strip()
                if name:
                    stats.trending_by_name[name] = (
                        stats.trending_by_name.get(name, 0) + 1
                    )
                continue
            hairstyle_id = card_hairstyle_id(card)
            if hairstyle_id is None or hairstyle_id in seen:
                continue
            seen.add(hairstyle_id)
            social_proof = card.get("engagement_reason") is True
            if social_proof:
                stats.social_proof_exposures += 1
            exposures.append(
                Exposure(
                    str(analysis_id),
                    hairstyle_id,
                    _card_rank(card),
                    group,
                    subject,
                    social_proof,
                )
            )
            stats.ml_exposures += 1

    stats.first_gender_at = first_gender.isoformat() if first_gender else None
    return exposures, stats


# ========== 반응 ==========
@dataclass(frozen=True)
class ReactionSample:
    analysis_id: str
    hairstyle_id: int
    kind: str


def sample_from_metadata(metadata: Mapping[str, Any]) -> Optional[ReactionSample]:
    """npz metadata(JSON) -> ReactionSample. 해석할 수 없으면 None"""
    analysis_id = metadata.get("analysis_id")
    hairstyle_id = card_hairstyle_id({"hairstyle_id": metadata.get("hairstyle_id")})
    if not analysis_id or hairstyle_id is None:
        return None

    if metadata.get("source") == "implicit":
        kind = metadata.get("signal_type")
        if kind not in IMPLICIT_KINDS:
            return None
        return ReactionSample(str(analysis_id), hairstyle_id, kind)

    feedback = metadata.get("feedback")
    if isinstance(feedback, str):
        normalized = feedback.strip().lower()
        if normalized in _POSITIVE_FEEDBACK:
            return ReactionSample(str(analysis_id), hairstyle_id, KIND_EXPLICIT_GOOD)
        if normalized in _NEGATIVE_FEEDBACK:
            return ReactionSample(str(analysis_id), hairstyle_id, KIND_EXPLICIT_BAD)
    return None


def read_npz_metadata(body: bytes) -> Optional[Dict[str, Any]]:
    """npz 바이트에서 metadata JSON 만 꺼낸다 (allow_pickle=False). 실패 시 None"""
    import numpy as np

    try:
        with np.load(io.BytesIO(body), allow_pickle=False) as data:
            if "metadata" not in data.files:
                return None
            raw = data["metadata"]
            text = str(raw[0]) if getattr(raw, "shape", ()) else str(raw)
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _stronger(a: Tuple[float, str], b: Tuple[float, str]) -> bool:
    """a 가 b 보다 강한 신호인가 (가중치, 같으면 신호 종류 순)"""
    return (a[0], _KIND_TIEBREAK.get(a[1], -1)) > (b[0], _KIND_TIEBREAK.get(b[1], -1))


def reduce_reactions(
    samples: Iterable[ReactionSample],
    weights: Mapping[str, float],
    include_explicit: bool = True,
) -> Dict[Tuple[str, int], Tuple[float, str]]:
    """(analysis_id, hairstyle_id) -> (가중치, 신호) - 쌍마다 가장 강한 것 1개

    pending 과 processed 에 같은 쌍이 함께 있어도(학습 후 새 신호) 하나로 합친다.
    """
    best: Dict[Tuple[str, int], Tuple[float, str]] = {}
    for sample in samples:
        if sample.kind in EXPLICIT_KINDS and not include_explicit:
            continue
        candidate = (float(weights.get(sample.kind, 0.0)), sample.kind)
        key = (sample.analysis_id, sample.hairstyle_id)
        current = best.get(key)
        if current is None or _stronger(candidate, current):
            best[key] = candidate
    return best


# ========== 계층적 축소 ==========
def shrunk_rate(
    reactions: float, exposures: float, prior_rate: float, m: float
) -> float:
    """(R + m * prior) / (N + m). 노출·사전 강도가 모두 0이면 prior 그대로"""
    denominator = exposures + m
    if denominator <= 0:
        return prior_rate
    return (reactions + m * prior_rate) / denominator


def prior_strength(a: float, mu: float) -> float:
    """반응 단위 사전 강도 a -> 노출 단위 m = a / mu (mu == 0 이면 0)"""
    if mu <= 0:
        return 0.0
    return a / mu


@dataclass
class _Unit:
    """주체 x 그룹 x 스타일 1개 (노출 1, 반응은 최댓값 1개)"""

    hairstyle_id: int
    group: Optional[str]
    subject: str
    rank: Optional[int]
    reaction: Tuple[float, str] = (0.0, "")


def collapse_units(
    exposures: List[Exposure],
    reactions: Mapping[Tuple[str, int], Tuple[float, str]],
) -> Tuple[List[_Unit], Dict[str, Any]]:
    """노출 -> 주체당 기여 상한을 적용한 단위. 사회적 증거 노출은 빼고 센다"""
    units: Dict[Tuple[str, Optional[str], int], _Unit] = {}
    matched: Set[Tuple[str, int]] = set()
    excluded_matched: Set[Tuple[str, int]] = set()
    matched_kinds: Dict[str, int] = defaultdict(int)
    social_proof = 0

    for e in exposures:
        pair = (e.analysis_id, e.hairstyle_id)
        reaction = reactions.get(pair)
        if e.social_proof:
            social_proof += 1
            if reaction is not None:
                excluded_matched.add(pair)
            continue
        if reaction is not None and pair not in matched:
            matched.add(pair)
            matched_kinds[reaction[1]] += 1
        subject = e.subject or f"analysis:{e.analysis_id}"
        key = (subject, e.group, e.hairstyle_id)
        unit = units.get(key)
        if unit is None:
            unit = units[key] = _Unit(e.hairstyle_id, e.group, subject, e.rank)
        elif e.rank is not None and (unit.rank is None or e.rank < unit.rank):
            unit.rank = e.rank
        if reaction is not None and _stronger(reaction, unit.reaction):
            unit.reaction = reaction

    orphan_kinds: Dict[str, int] = defaultdict(int)
    for pair, (_, kind) in reactions.items():
        if pair not in matched and pair not in excluded_matched:
            orphan_kinds[kind] += 1

    exposures_in = len(exposures) - social_proof
    return list(units.values()), {
        "exposures": exposures_in,
        "units": len(units),
        "capped_exposures": exposures_in - len(units),
        "social_proof_exposures": social_proof,
        "social_proof_reactions": len(excluded_matched),
        "matched_pairs": len(matched),
        "matched_by_kind": dict(sorted(matched_kinds.items())),
        "orphan_pairs": sum(orphan_kinds.values()),
        "orphan_by_kind": dict(sorted(orphan_kinds.items())),
    }


def position_factors(
    units: List[_Unit], prior: float, iterations: int = 30
) -> Dict[int, float]:
    """스타일 고정효과를 뺀 위치 효과 f_rank (둘 이상 순위에 노출된 스타일만 사용)

    rho_s (스타일 반응률) 와 f_r (위치 배수) 를 번갈아 맞춘다:
        rho_s = R_s / sum_r n_sr f_r,   f_r = R_r / sum_s n_sr rho_s
    노출 가중 평균이 1 이 되도록 맞춘 뒤 (R_r + prior) / (E_r + prior) 로 1 에 축소.
    같은 스타일이 여러 순위에 나온 데이터가 없으면 {} (보정 없음).
    """
    ranks_by_style: Dict[int, Set[int]] = defaultdict(set)
    for u in units:
        if u.rank is not None:
            ranks_by_style[u.hairstyle_id].add(u.rank)
    eligible = {s for s, ranks in ranks_by_style.items() if len(ranks) >= 2}

    n: Dict[Tuple[int, int], int] = defaultdict(int)
    r: Dict[Tuple[int, int], float] = defaultdict(float)
    for u in units:
        if u.rank is None or u.hairstyle_id not in eligible:
            continue
        n[(u.hairstyle_id, u.rank)] += 1
        r[(u.hairstyle_id, u.rank)] += u.reaction[0]
    if not n or sum(r.values()) <= 0:
        return {}

    styles = sorted({s for s, _ in n})
    ranks = sorted({k for _, k in n})
    f = {k: 1.0 for k in ranks}
    rho = {s: 0.0 for s in styles}
    react_s = {s: sum(r[(s, k)] for k in ranks) for s in styles}
    react_r = {k: sum(r[(s, k)] for s in styles) for k in ranks}
    count_r = {k: sum(n[(s, k)] for s in styles) for k in ranks}

    for _ in range(iterations):
        for s in styles:
            denom = sum(n[(s, k)] * f[k] for k in ranks)
            rho[s] = react_s[s] / denom if denom > 0 else 0.0
        for k in ranks:
            expected = sum(n[(s, k)] * rho[s] for s in styles)
            f[k] = react_r[k] / expected if expected > 0 else 1.0
        total = sum(count_r.values())
        mean_f = sum(count_r[k] * f[k] for k in ranks) / total
        if mean_f > 0:
            f = {k: v / mean_f for k, v in f.items()}

    result = {}
    for k in ranks:
        expected = react_r[k] / f[k] if f[k] > 0 else react_r[k]
        result[k] = (react_r[k] + prior) / (expected + prior)
    return result


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def compute_engagement(
    exposures: List[Exposure],
    reactions: Mapping[Tuple[str, int], Tuple[float, str]],
    *,
    a1: float = DEFAULT_A1,
    a2: float = DEFAULT_A2,
    ag: float = DEFAULT_AG,
    position_adjust: bool = DEFAULT_POSITION_ADJUST,
    position_prior: float = DEFAULT_POSITION_PRIOR,
) -> Dict[str, Any]:
    """노출·반응 -> {mu, m1, m2, position_factors, styles, groups, group_totals, summary}

    styles / groups 의 키는 문자열 hairstyle_id (JSON 키).
    노출이 없는 스타일·그룹 쌍은 넣지 않는다 (서빙에서 상위 단계 값으로 대체).
    """
    units, summary = collapse_units(exposures, reactions)
    factors = position_factors(units, position_prior) if position_adjust else {}

    style_n: Dict[int, float] = defaultdict(float)
    style_raw_n: Dict[int, int] = defaultdict(int)
    style_r: Dict[int, float] = defaultdict(float)
    style_subjects: Dict[int, Set[str]] = defaultdict(set)
    cell_n: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    cell_raw_n: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    cell_r: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    cell_subjects: Dict[str, Dict[int, Set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for u in units:
        effective = factors.get(u.rank, 1.0) if u.rank is not None else 1.0
        weight, kind = u.reaction
        save_like = kind in SAVE_LIKE_KINDS
        style_n[u.hairstyle_id] += effective
        style_raw_n[u.hairstyle_id] += 1
        style_r[u.hairstyle_id] += weight
        if save_like:
            style_subjects[u.hairstyle_id].add(u.subject)
        if u.group is not None:
            cell_n[u.group][u.hairstyle_id] += effective
            cell_raw_n[u.group][u.hairstyle_id] += 1
            cell_r[u.group][u.hairstyle_id] += weight
            if save_like:
                cell_subjects[u.group][u.hairstyle_id].add(u.subject)

    total_n = sum(style_n.values())
    total_r = sum(style_r.values())
    mu = total_r / total_n if total_n > 0 else 0.0
    m1 = prior_strength(a1, mu)
    m2 = prior_strength(a2, mu)
    mg = prior_strength(ag, mu)

    style_rates: Dict[int, float] = {}
    styles: Dict[str, Dict[str, Any]] = {}
    for hid in sorted(style_n):
        rate = shrunk_rate(style_r[hid], style_n[hid], mu, m1)
        style_rates[hid] = rate
        styles[str(hid)] = {
            "rate": _round(rate),
            "exposures": style_raw_n[hid],
            "effective_exposures": _round(style_n[hid], 3),
            "reactions": _round(style_r[hid], 3),
            "subjects": len(style_subjects[hid]),
            "auth_subjects": sum(map(is_auth_subject, style_subjects[hid])),
        }

    groups: Dict[str, Dict[str, Dict[str, Any]]] = {}
    group_totals: Dict[str, Dict[str, Any]] = {}
    for gkey in sorted(cell_n):
        group_reactions = sum(cell_r[gkey].values())
        group_exposures = sum(cell_n[gkey].values())
        group_mean = shrunk_rate(group_reactions, group_exposures, mu, mg)
        scale = group_mean / mu if mu > 0 else 1.0
        entries: Dict[str, Dict[str, Any]] = {}
        for hid in sorted(cell_n[gkey]):
            rate = shrunk_rate(
                cell_r[gkey][hid], cell_n[gkey][hid], style_rates[hid] * scale, m2
            )
            # lift 기준선: 이 스타일을 뺀 그룹 평균 (leave-one-out). 자주 나오는 스타일의
            # 반응이 자기 기준선을 끌어올려 lift 를 깎는 것을 막는다
            base = shrunk_rate(
                group_reactions - cell_r[gkey][hid],
                group_exposures - cell_n[gkey][hid],
                mu,
                mg,
            )
            entries[str(hid)] = {
                "rate": _round(rate),
                "base": _round(base),
                "exposures": cell_raw_n[gkey][hid],
                "effective_exposures": _round(cell_n[gkey][hid], 3),
                "reactions": _round(cell_r[gkey][hid], 3),
                "subjects": len(cell_subjects[gkey][hid]),
                "auth_subjects": sum(map(is_auth_subject, cell_subjects[gkey][hid])),
            }
        groups[gkey] = entries
        group_totals[gkey] = {
            "exposures": sum(cell_raw_n[gkey].values()),
            "reactions": _round(group_reactions, 3),
            "styles": len(entries),
            "mean_rate": _round(group_mean),
        }

    summary.update(
        {
            "effective_exposures": _round(total_n, 3),
            "reaction_weight": _round(total_r, 3),
        }
    )
    return {
        "mu": _round(mu),
        "m1": _round(m1, 3),
        "m2": _round(m2, 3),
        "mg": _round(mg, 3),
        "position_factors": {str(k): _round(v, 4) for k, v in factors.items()},
        "styles": styles,
        "groups": groups,
        "group_totals": group_totals,
        "summary": summary,
    }


def build_payload(
    exposures: List[Exposure],
    reactions: Mapping[Tuple[str, int], Tuple[float, str]],
    *,
    since: datetime,
    until: datetime,
    weights: Mapping[str, float],
    include_explicit: bool,
    a1: float = DEFAULT_A1,
    a2: float = DEFAULT_A2,
    ag: float = DEFAULT_AG,
    position_adjust: bool = DEFAULT_POSITION_ADJUST,
    position_prior: float = DEFAULT_POSITION_PRIOR,
    face_shape_since: Optional[datetime] = None,
    generated_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """S3 engagement/current.json 형식

    version 은 생성 시각(UTC) 기반 문자열 - 서빙이 meta.engagement_version 으로 남긴다.
    """
    generated = generated_at or datetime.now(timezone.utc)
    result = compute_engagement(
        exposures,
        reactions,
        a1=a1,
        a2=a2,
        ag=ag,
        position_adjust=position_adjust,
        position_prior=position_prior,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "version": generated.strftime("eng_%Y%m%dT%H%M%SZ"),
        "generated_at": generated.isoformat(),
        "window": {"since": since.isoformat(), "until": until.isoformat()},
        "mu": result["mu"],
        "params": {
            "a1": a1,
            "a2": a2,
            "ag": ag,
            "m1": result["m1"],
            "m2": result["m2"],
            "mg": result["mg"],
            "weights": dict(weights),
            "include_explicit": include_explicit,
            "subject_cap": "subject x group x style: exposure 1, max reaction 1",
            "position_adjust": position_adjust,
            "position_prior": position_prior,
            "position_factors": result["position_factors"],
            "face_shape_since": (
                face_shape_since.isoformat() if face_shape_since else None
            ),
        },
        "summary": result["summary"],
        "styles": result["styles"],
        "groups": result["groups"],
        "group_totals": result["group_totals"],
    }
