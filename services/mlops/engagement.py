"""스타일 반응률 집계 (추천 보너스용)

추천 모델 점수에 "같은 얼굴 그룹(성별 x 얼굴형)에서 이 스타일이 화면에 나갔을 때
얼마나 반응을 얻었는가" 를 보너스로 더하기 위한 통계를 만든다.
서빙 쪽은 services/rec_engagement.py, 정기 실행은 scripts/build_style_engagement.py.

노출 (exposure)
    hairme-analysis 의 분석별 카드 중 화면에 나간 ML 카드(source == "ml").
    트렌드 카드는 노출 확률·hairstyle_id 가 없어 개수만 따로 센다.

반응 (reaction)
    S3 hairme-mlops 의 학습 샘플 npz (feedback/pending/** 와 트레이너가 옮긴
    feedback/processed/**). 암묵 신호는 metadata.source == "implicit" 의
    signal_type, 명시 피드백은 metadata.feedback (good/bad).
    analysis_id x hairstyle_id 쌍마다 가중치가 가장 큰 신호 1개만 쓴다.
    노출이 없는 쌍(창 밖 분석, 트렌드 카드, 노출 기록 이전 데이터)은 버리고 개수만 보고.

위치 보정
    1위 카드는 3위보다 반응을 더 받는다 (2026-09~10 명시 피드백: 1위 15 / 2위 8 / 3위 3).
    보정이 없으면 늘 1위에 나오던 스타일이 위치 덕에 보너스를 받아 편중이 더 심해진다.
    그래서 노출 1건을 "그 위치의 기대 반응 / 전체 평균" (f_rank) 만큼으로 센다.
    f_rank 는 표본이 적을 때 1 로 축소한다.

계층적 축소 (반응률 = 반응 가중합 / 유효 노출)
    mu        = 전체 반응합 / 전체 유효 노출
    style     = (R_s + m1 * mu) / (N_s + m1)
    group     = (R_gs + m2 * style) / (N_gs + m2)
    m1 = a1 / mu, m2 = a2 / mu  (a = "반응 가중치 단위" 사전 강도)

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

출력 JSON 에는 hairstyle_id·그룹 키·숫자만 넣는다 (user_id / device_id / analysis_id 없음).
"""

from __future__ import annotations

import io
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

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

DEFAULT_WEIGHTS: Dict[str, float] = {
    KIND_SYNTHESIZE: 1.0,
    KIND_SAVE: 3.0,
    KIND_SHARE: 4.0,
    KIND_SHOWN_NOT_CHOSEN: 0.0,
    KIND_EXPLICIT_GOOD: 2.0,
    KIND_EXPLICIT_BAD: 0.0,
}

# 같은 가중치일 때 어느 신호로 기록할지 (보고용 - 반응합에는 영향 없음)
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
DEFAULT_A2 = 20.0  # 그룹 반응률 사전 강도 (반응 가중치 단위)
DEFAULT_POSITION_PRIOR = 20.0  # 위치 보정 f_rank 사전 강도 (반응 가중치 단위)
FACE_SHAPE_RELIABLE_SINCE = "2026-09-17"

UNKNOWN_GENDER = "unknown"
KNOWN_GENDERS = ("male", "female")

CARD_SOURCE_ML = "ml"


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
    """

    analysis_id: str
    hairstyle_id: int
    rank: Optional[int]
    group: Optional[str]


@dataclass
class ExposureStats:
    analyses: int = 0
    ml_exposures: int = 0
    trending_exposures: int = 0
    skipped_out_of_window: int = 0
    skipped_no_cards: int = 0
    gender_unknown_analyses: int = 0
    face_shape_untrusted_analyses: int = 0
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
            "skipped_out_of_window": self.skipped_out_of_window,
            "skipped_no_cards": self.skipped_no_cards,
            "gender_unknown_analyses": self.gender_unknown_analyses,
            "face_shape_untrusted_analyses": self.face_shape_untrusted_analyses,
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
            exposures.append(
                Exposure(str(analysis_id), hairstyle_id, _card_rank(card), group)
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


def reduce_reactions(
    samples: Iterable[ReactionSample],
    weights: Mapping[str, float],
    include_explicit: bool = True,
) -> Dict[Tuple[str, int], Tuple[float, str]]:
    """(analysis_id, hairstyle_id) -> (가중치, 신호) - 쌍마다 가장 강한 것 1개"""
    best: Dict[Tuple[str, int], Tuple[float, str]] = {}
    for sample in samples:
        if sample.kind in EXPLICIT_KINDS and not include_explicit:
            continue
        weight = float(weights.get(sample.kind, 0.0))
        key = (sample.analysis_id, sample.hairstyle_id)
        current = best.get(key)
        candidate = (weight, sample.kind)
        if current is None or (weight, _KIND_TIEBREAK.get(sample.kind, -1)) > (
            current[0],
            _KIND_TIEBREAK.get(current[1], -1),
        ):
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


def position_factors(
    exposures: List[Exposure],
    reactions: Mapping[Tuple[str, int], Tuple[float, str]],
    prior: float,
) -> Dict[int, float]:
    """위치별 기대 반응 비율 f_rank = (R_r + prior) / (N_r * mu + prior)

    표본이 적으면 1 로 수렴한다. rank 가 없는 노출은 1.0.
    """
    total_n = len(exposures)
    total_r = sum(
        reactions.get((e.analysis_id, e.hairstyle_id), (0.0, ""))[0] for e in exposures
    )
    if total_n == 0 or total_r <= 0:
        return {}
    mu = total_r / total_n
    count: Dict[int, int] = defaultdict(int)
    react: Dict[int, float] = defaultdict(float)
    for e in exposures:
        if e.rank is None:
            continue
        count[e.rank] += 1
        react[e.rank] += reactions.get((e.analysis_id, e.hairstyle_id), (0.0, ""))[0]
    return {
        rank: (react[rank] + prior) / (count[rank] * mu + prior)
        for rank in sorted(count)
    }


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def compute_engagement(
    exposures: List[Exposure],
    reactions: Mapping[Tuple[str, int], Tuple[float, str]],
    *,
    a1: float = DEFAULT_A1,
    a2: float = DEFAULT_A2,
    position_adjust: bool = True,
    position_prior: float = DEFAULT_POSITION_PRIOR,
) -> Dict[str, Any]:
    """노출·반응 -> {mu, m1, m2, position_factors, styles, groups, group_totals, summary}

    styles / groups 의 키는 문자열 hairstyle_id (JSON 키).
    노출이 없는 스타일·그룹 쌍은 넣지 않는다 (서빙에서 상위 단계 값으로 대체).
    """
    factors = (
        position_factors(exposures, reactions, position_prior)
        if position_adjust
        else {}
    )

    style_n: Dict[int, float] = defaultdict(float)
    style_raw_n: Dict[int, int] = defaultdict(int)
    style_r: Dict[int, float] = defaultdict(float)
    group_n: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    group_raw_n: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    group_r: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    kind_counts: Dict[str, int] = defaultdict(int)
    matched = set()

    for e in exposures:
        effective = factors.get(e.rank, 1.0) if e.rank is not None else 1.0
        key = (e.analysis_id, e.hairstyle_id)
        weight, kind = reactions.get(key, (0.0, ""))
        if key in reactions and key not in matched:
            matched.add(key)
            kind_counts[kind] += 1
        style_n[e.hairstyle_id] += effective
        style_raw_n[e.hairstyle_id] += 1
        style_r[e.hairstyle_id] += weight
        if e.group is not None:
            group_n[e.group][e.hairstyle_id] += effective
            group_raw_n[e.group][e.hairstyle_id] += 1
            group_r[e.group][e.hairstyle_id] += weight

    total_n = sum(style_n.values())
    total_r = sum(style_r.values())
    mu = total_r / total_n if total_n > 0 else 0.0
    m1 = prior_strength(a1, mu)
    m2 = prior_strength(a2, mu)

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
        }

    groups: Dict[str, Dict[str, Dict[str, Any]]] = {}
    group_totals: Dict[str, Dict[str, Any]] = {}
    for gkey in sorted(group_n):
        entries: Dict[str, Dict[str, Any]] = {}
        for hid in sorted(group_n[gkey]):
            rate = shrunk_rate(
                group_r[gkey][hid], group_n[gkey][hid], style_rates[hid], m2
            )
            entries[str(hid)] = {
                "rate": _round(rate),
                "exposures": group_raw_n[gkey][hid],
                "effective_exposures": _round(group_n[gkey][hid], 3),
                "reactions": _round(group_r[gkey][hid], 3),
            }
        groups[gkey] = entries
        group_totals[gkey] = {
            "exposures": sum(group_raw_n[gkey].values()),
            "reactions": _round(sum(group_r[gkey].values()), 3),
            "styles": len(entries),
        }

    orphan_kinds: Dict[str, int] = defaultdict(int)
    for key, (_, kind) in reactions.items():
        if key not in matched:
            orphan_kinds[kind] += 1

    return {
        "mu": _round(mu),
        "m1": _round(m1, 3),
        "m2": _round(m2, 3),
        "position_factors": {str(k): _round(v, 4) for k, v in factors.items()},
        "styles": styles,
        "groups": groups,
        "group_totals": group_totals,
        "summary": {
            "exposures": len(exposures),
            "effective_exposures": _round(total_n, 3),
            "reaction_weight": _round(total_r, 3),
            "matched_pairs": len(matched),
            "matched_by_kind": dict(sorted(kind_counts.items())),
            "orphan_pairs": sum(orphan_kinds.values()),
            "orphan_by_kind": dict(sorted(orphan_kinds.items())),
        },
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
    position_adjust: bool = True,
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
            "m1": result["m1"],
            "m2": result["m2"],
            "weights": dict(weights),
            "include_explicit": include_explicit,
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
