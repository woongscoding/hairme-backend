# -*- coding: utf-8 -*-
"""추천 반응률 보너스 (engagement_v1) 테스트

- 집계: 계층 축소 공식, 노출 0, 중복 쌍(pending+processed), 주체당 기여 상한,
  그룹 기저율 기준 lift, 위치 보정(스타일 고정효과), 사회적 증거 노출 제외,
  save/share 고유 주체 수, Decimal/str hairstyle_id, 얼굴형 재보정 이전 행
- 서빙: 셀 -> 스타일 -> 0 폴백, 보너스 상한, TTL 캐시, stale-while-error, 신선도 한도,
  S3 실패·검증 실패 시 보너스 0 + engagement_table_rejected 로그
- 플래그 꺼짐 / 대조군 / 테이블 없음 응답이 기존(f3aa191) 스냅샷과 바이트 단위 동일
- 탐색 x 보너스 4가지 조합의 호출 k·순위·동점권·meta, ITT(engagement_applied)
- 사유 문구 조건(save/share 고유 주체), engagement_reason 카드 표식
- 엔드포인트 저장·로그 필드, 집계 스크립트(--write 가드)
- 공격 회귀: 같은 image_hash 반복 share 가 셀을 못 움직임, 표식 없는 save 무시

AWS 없이 돈다 (S3 로더·recommend_top_k 는 가짜).
"""

import io
import json
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

import numpy as np
import pytest

from config.settings import settings
from services import rec_engagement
from services.mlops import engagement as eng
from services.rec_engagement import (
    ARM_CONTROL,
    ARM_TREATMENT,
    EngagementStore,
    EngagementTable,
    TableRejected,
    compute_bonus,
)
from services.rec_exploration import ARM_CONTROL as EXPLORE_CONTROL
from services.rec_exploration import ARM_EXPLORE, POLICY_DETERMINISTIC, POLICY_EXPLORE

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "rec_engagement_baseline.json"

MODEL_VERSION = "v6_feedback_20260909"
IMAGE_HASH = "e" * 64

# 스냅샷 생성 때와 같은 풀 (hairstyle_id = 100 + i)
SCORES = [
    62.4, 62.1, 61.9, 61.7, 61.2, 60.8, 60.5, 60.1, 59.6, 59.0,
    58.2, 57.5, 56.9, 56.0, 55.1, 54.3, 53.0, 52.2, 51.0, 50.0,
]  # fmt: skip
TRENDING = [
    {"style_name": "트렌드A", "reason": "요즘 인기", "source": "trending"},
    {"style_name": "트렌드B", "reason": "요즘 인기", "source": "trending"},
]
NOW = datetime(2026, 11, 2, tzinfo=timezone.utc)
G = "male|둥근형"


def _ranked(scores=SCORES):
    return [
        {
            "hairstyle_id": 100 + i,
            "hairstyle": f"스타일{i:02d}",
            "score": round(s, 2),
            "original_score": s,
        }
        for i, s in enumerate(scores)
    ]


def _service(ranked=None):
    from services.hybrid_recommender import MLRecommendationService

    ranked = ranked if ranked is not None else _ranked()
    with patch.object(MLRecommendationService, "__init__", return_value=None):
        service = MLRecommendationService()
    service.ml_available = True
    service.reason_generator = None
    service.ml_recommender = MagicMock()
    service.ml_recommender.allowed_indices = None
    service.ml_recommender.model_version = MODEL_VERSION
    service.ml_recommender.recommend_top_k.side_effect = lambda **kw: [
        dict(r) for r in ranked[: kw["k"]]
    ]
    return service


def _recommend(
    service, image_hash=IMAGE_HASH, gender="male", face_shape="둥근형", out=None
):
    with patch(
        "services.trending_style_service.get_trending_style_service"
    ) as get_trending:
        get_trending.return_value.pick_trending.return_value = [
            dict(t) for t in TRENDING
        ]
        return service.recommend(
            image_data=b"",
            face_shape=face_shape,
            skin_tone="봄웜",
            face_features=[1.4, 120.0, 130.0, 110.0, 0.92, 0.85],
            skin_features=[45.0, 15.0],
            gender=gender,
            image_hash=image_hash,
            user_id=None,
            engagement_out=out,
        )


def _ml_cards(result):
    return [r for r in result["recommendations"] if r.get("source") == "ml"]


def _payload(
    mu=0.5,
    styles=None,
    groups=None,
    version="eng_test_v1",
    group_totals=None,
    generated_at=None,
):
    payload = {
        "schema_version": 1,
        "version": version,
        "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(),
        "mu": mu,
        "styles": styles if styles is not None else {"1": _entry(mu)},
        "groups": groups or {},
    }
    if group_totals is not None:
        payload["group_totals"] = group_totals
    return payload


def _entry(rate, exposures=100, reactions=0.0, subjects=0):
    return {
        "rate": rate,
        "exposures": exposures,
        "reactions": reactions,
        "subjects": subjects,
    }


class _FakeStore:
    def __init__(self, table):
        self.table = table
        self.calls = 0

    def get(self):
        self.calls += 1
        return self.table


@pytest.fixture
def base_settings(monkeypatch):
    monkeypatch.setattr(settings, "REC_EXPLORE_ENABLED", False)
    monkeypatch.setattr(settings, "REC_EXPLORE_RATIO", 0.5)
    monkeypatch.setattr(settings, "REC_EXPLORE_POOL", 20)
    monkeypatch.setattr(settings, "REC_EXPLORE_MARGIN", 3.0)
    monkeypatch.setattr(settings, "REC_EXPLORE_TEMPERATURE", 1.5)
    monkeypatch.setattr(settings, "FULL_RANKING_LIMIT", 80)
    monkeypatch.setattr(settings, "REC_ENGAGEMENT_ENABLED", False)
    monkeypatch.setattr(settings, "REC_ENGAGEMENT_RATIO", 1.0)
    monkeypatch.setattr(settings, "REC_ENGAGEMENT_BETA", 3.0)
    monkeypatch.setattr(settings, "REC_ENGAGEMENT_CAP", 3.0)
    monkeypatch.setattr(settings, "REC_ENGAGEMENT_REASON_MIN_LIFT", 0.3)
    monkeypatch.setattr(settings, "REC_ENGAGEMENT_REASON_MIN_EXPOSURES", 50)
    monkeypatch.setattr(settings, "REC_ENGAGEMENT_REASON_MIN_SUBJECTS", 15)
    yield
    rec_engagement.reset_engagement_store(None)


def _install_table(payload):
    table = EngagementTable.from_payload(payload) if payload is not None else None
    store = _FakeStore(table)
    rec_engagement.reset_engagement_store(store)  # type: ignore[arg-type]
    return store


# 보너스로 스타일05(105)를 끌어올리고 스타일00(100)을 내리는 테이블 (그룹 평균 = mu)
BOOST_PAYLOAD = _payload(
    mu=0.5,
    styles={"100": _entry(0.5), "105": _entry(0.6)},
    groups={
        G: {
            "100": _entry(0.2, 300),  # lift -0.6 -> -1.8
            "105": _entry(1.0, 120, 60.0, 20),  # lift +1.0 -> +3.0 (상한)
        }
    },
)


def _exp(aid, hid, rank=1, group=G, subject=None, social_proof=False):
    return eng.Exposure(aid, hid, rank, group, subject, social_proof)


# ========== 집계: 축소 공식 ==========


class TestShrinkage:
    def test_shrunk_rate_formula(self):
        assert eng.shrunk_rate(10.0, 20.0, 0.5, 20.0) == pytest.approx((10 + 10) / 40)

    def test_zero_exposure_returns_prior(self):
        assert eng.shrunk_rate(0.0, 0.0, 0.37, 40.0) == pytest.approx(0.37)
        assert eng.shrunk_rate(0.0, 0.0, 0.37, 0.0) == pytest.approx(0.37)

    def test_prior_strength_scales_with_mu(self):
        assert eng.prior_strength(20.0, 0.5) == pytest.approx(40.0)
        assert eng.prior_strength(20.0, 0.01) == pytest.approx(2000.0)
        assert eng.prior_strength(20.0, 0.0) == 0.0

    def test_hierarchical_rates(self):
        exposures = [_exp(f"a{i}", 1) for i in range(10)] + [
            _exp(f"b{i}", 2, group=None) for i in range(10)
        ]
        reactions = {("a0", 1): (3.0, "save"), ("a1", 1): (1.0, "synthesize")}
        result = eng.compute_engagement(exposures, reactions, a1=2.0, a2=1.0, ag=1.0)
        mu = 4.0 / 20
        m1, m2, mg = 2.0 / mu, 1.0 / mu, 1.0 / mu
        style1 = (4.0 + m1 * mu) / (10 + m1)
        style2 = (0.0 + m1 * mu) / (10 + m1)
        group_mean = (4.0 + mg * mu) / (10 + mg)
        cell_prior = style1 * group_mean / mu
        assert result["mu"] == pytest.approx(mu)
        assert result["m1"] == pytest.approx(m1)
        assert result["styles"]["1"]["rate"] == pytest.approx(style1, abs=1e-6)
        assert result["styles"]["2"]["rate"] == pytest.approx(style2, abs=1e-6)
        assert result["group_totals"][G]["mean_rate"] == pytest.approx(
            group_mean, abs=1e-6
        )
        cell = result["groups"][G]["1"]
        assert cell["rate"] == pytest.approx(
            (4.0 + m2 * cell_prior) / (10 + m2), abs=1e-6
        )
        assert cell["exposures"] == 10
        assert cell["subjects"] == 1  # save 1명 (합성은 세지 않는다)
        # group=None 노출은 스타일 통계에만
        assert "2" not in result["groups"][G]
        assert result["group_totals"][G]["exposures"] == 10

    def test_no_reactions_gives_mu_zero(self):
        result = eng.compute_engagement([_exp("a", 1)], {})
        assert result["mu"] == 0.0
        assert result["m1"] == 0.0
        assert result["styles"]["1"]["rate"] == 0.0


class TestGroupBaseline:
    def _world(self, hi_group_rate):
        """두 그룹, 10개 스타일 동일 품질. 한 그룹만 반응 기저율이 높다"""
        import random

        rng = random.Random(1)
        exposures, reactions = [], {}
        for group, p in (("male|둥근형", 0.3), ("female|계란형", hi_group_rate)):
            for i in range(600):
                aid = f"{group}-{i}"
                for rank, hid in enumerate(rng.sample(range(10), 3), 1):
                    exposures.append(_exp(aid, hid, rank, group))
                    if rng.random() < p:
                        reactions[(aid, hid)] = (1.0, "synthesize")
        payload = eng.build_payload(
            exposures,
            reactions,
            since=NOW - timedelta(days=60),
            until=NOW,
            weights=eng.DEFAULT_WEIGHTS,
            include_explicit=True,
        )
        return EngagementTable.from_payload(payload)

    def test_high_baseline_group_does_not_get_blanket_bonus(self):
        table = self._world(hi_group_rate=0.6)
        hi = [
            compute_bonus(table.lift("female|계란형", h), 3.0, 3.0) for h in range(10)
        ]
        normal = [
            compute_bonus(table.lift("male|둥근형", h), 3.0, 3.0) for h in range(10)
        ]
        # 그룹 평균 기준이라 전부 + 가 되지 않는다 (예전 mu 기준이면 전부 약 +1)
        assert sum(1 for b in hi if b > 0) < 9
        assert abs(sum(hi) / 10) < 0.4
        assert abs(sum(normal) / 10) < 0.4
        assert table.group_mean("female|계란형") > 1.5 * table.group_mean("male|둥근형")

    def test_dominant_style_baseline_excludes_itself(self):
        """그룹 노출 대부분을 차지하는 스타일의 반응이 자기 기준선을 끌어올리지 않는다"""
        exposures, reactions = [], {}
        for i in range(2000):
            hid = 1 if i % 5 else 2 + i % 3  # 80% 가 스타일 1
            exposures.append(_exp(f"a{i}", hid, subject=f"s{i}"))
            p = 0.6 if hid == 1 else 0.3
            if (i * 7919) % 1000 < p * 1000:
                reactions[(f"a{i}", hid)] = (1.0, "synthesize")
        payload = eng.build_payload(
            exposures,
            reactions,
            since=NOW - timedelta(days=60),
            until=NOW,
            weights=eng.DEFAULT_WEIGHTS,
            include_explicit=True,
        )
        cell = payload["groups"][G]["1"]
        assert cell["base"] == pytest.approx(0.3, abs=0.06)
        table = EngagementTable.from_payload(payload)
        assert table.lift(G, 1) == pytest.approx(
            1.0, abs=0.25
        )  # 그룹 평균 기준이면 ~0.2

    def test_missing_cell_falls_back_to_style_lift(self):
        table = EngagementTable.from_payload(
            _payload(
                mu=0.5,
                styles={"7": _entry(0.75)},
                groups={G: {"1": _entry(1.0)}},
                group_totals={G: {"mean_rate": 1.0}},
            )
        )
        assert table.lift(G, 1) == pytest.approx(0.0)  # 셀 1.0 / 그룹 평균 1.0
        assert table.lift(G, 7) == pytest.approx(0.5)  # 셀 없음 -> 스타일 / mu
        assert table.lift(G, 999) == 0.0


# ========== 집계: 위치 보정 ==========


class TestPositionFactors:
    def test_default_is_off(self):
        assert eng.DEFAULT_POSITION_ADJUST is False
        exposures = [_exp(f"a{i}", i % 3, 1 + i % 3) for i in range(30)]
        reactions = {(f"a{i}", i % 3): (1.0, "synthesize") for i in range(0, 30, 3)}
        assert eng.compute_engagement(exposures, reactions)["position_factors"] == {}

    def _units(self, rows):
        return [
            eng._Unit(hid, G, f"s{i}", rank, (1.0, "synthesize") if hit else (0.0, ""))
            for i, (hid, rank, hit) in enumerate(rows)
        ]

    def test_confounded_ranks_are_not_corrected(self):
        """스타일마다 늘 같은 순위에만 나오면 위치와 품질을 구분할 수 없다 -> 보정 없음

        (예전 방식은 이 경우 품질 차이를 위치 효과로 보고 과보정했다)
        """
        rows = []
        for i in range(300):
            rank = 1 + i % 3
            rows.append((rank, rank, (i // 3) % 10 < {1: 3, 2: 2, 3: 1}[rank]))
        assert eng.position_factors(self._units(rows), prior=1.0) == {}

    def test_recovers_position_effect_separately_from_quality(self):
        import random

        rng = random.Random(7)
        quality = {1: 0.4, 2: 0.2, 3: 0.1}  # 스타일 품질 (4배 차이)
        position = {1: 1.5, 2: 1.0, 3: 0.5}
        rows = []
        for _ in range(6000):
            # 좋은 스타일이 1위에 더 자주 나오지만 모든 순위에 나온다
            hid = rng.choices([1, 2, 3], weights=[3, 2, 1])[0]
            rank = rng.choices([1, 2, 3], weights=[3 if hid == 1 else 1, 1, 1])[0]
            rows.append((hid, rank, rng.random() < quality[hid] * position[rank]))
        f = eng.position_factors(self._units(rows), prior=1.0)
        assert f[1] / f[3] == pytest.approx(3.0, rel=0.2)
        assert f[2] / f[3] == pytest.approx(2.0, rel=0.2)


# ========== 집계: 반응·주체·사회적 증거 ==========


class TestReactions:
    def test_duplicate_pair_keeps_strongest(self):
        samples = [
            eng.ReactionSample("a", 1, eng.KIND_SYNTHESIZE),
            eng.ReactionSample("a", 1, eng.KIND_SAVE),
            eng.ReactionSample("a", 1, eng.KIND_SHOWN_NOT_CHOSEN),
            eng.ReactionSample("a", 2, eng.KIND_EXPLICIT_GOOD),
            eng.ReactionSample("a", 2, eng.KIND_SYNTHESIZE),
            eng.ReactionSample("b", 1, eng.KIND_EXPLICIT_BAD),
        ]
        reduced = eng.reduce_reactions(samples, eng.DEFAULT_WEIGHTS)
        assert reduced[("a", 1)] == (3.0, eng.KIND_SAVE)
        assert reduced[("a", 2)] == (2.0, eng.KIND_EXPLICIT_GOOD)
        assert reduced[("b", 1)] == (0.0, eng.KIND_EXPLICIT_BAD)

    def test_share_weight_equals_save(self):
        """앱은 공유 시트를 열기만 해도 share 를 보낸다 -> save 보다 강하지 않다"""
        assert eng.DEFAULT_WEIGHTS[eng.KIND_SHARE] == eng.DEFAULT_WEIGHTS[eng.KIND_SAVE]
        reduced = eng.reduce_reactions(
            [
                eng.ReactionSample("a", 1, eng.KIND_SAVE),
                eng.ReactionSample("a", 1, eng.KIND_SHARE),
            ],
            eng.DEFAULT_WEIGHTS,
        )
        assert reduced[("a", 1)] == (3.0, eng.KIND_SHARE)

    def test_pending_and_processed_same_pair(self):
        """학습 후 processed 에 synthesize, 새로 pending 에 save 가 있는 쌍 -> save 1개"""
        metas = [
            {"analysis_id": "a", "hairstyle_id": 1, "source": "implicit",
             "signal_type": "synthesize"},  # processed
            {"analysis_id": "a", "hairstyle_id": "1", "source": "implicit",
             "signal_type": "save"},  # pending
        ]  # fmt: skip
        samples = [eng.sample_from_metadata(m) for m in metas]
        reduced = eng.reduce_reactions(samples, eng.DEFAULT_WEIGHTS)
        assert reduced == {("a", 1): (3.0, "save")}

    def test_explicit_can_be_excluded(self):
        samples = [
            eng.ReactionSample("a", 2, eng.KIND_EXPLICIT_GOOD),
            eng.ReactionSample("a", 2, eng.KIND_SYNTHESIZE),
        ]
        reduced = eng.reduce_reactions(
            samples, eng.DEFAULT_WEIGHTS, include_explicit=False
        )
        assert reduced == {("a", 2): (1.0, eng.KIND_SYNTHESIZE)}

    def test_sample_from_metadata(self):
        implicit = eng.sample_from_metadata(
            {"analysis_id": "x", "hairstyle_id": 5, "source": "implicit",
             "signal_type": "share"}
        )  # fmt: skip
        assert implicit == eng.ReactionSample("x", 5, eng.KIND_SHARE)
        explicit = eng.sample_from_metadata(
            {"analysis_id": "x", "hairstyle_id": "7", "feedback": "dislike"}
        )
        assert explicit == eng.ReactionSample("x", 7, eng.KIND_EXPLICIT_BAD)
        assert (
            eng.sample_from_metadata({"analysis_id": "x", "feedback": "good"}) is None
        )
        assert (
            eng.sample_from_metadata(
                {"analysis_id": "x", "hairstyle_id": 1, "source": "implicit",
                 "signal_type": "?"}
            )  # fmt: skip
            is None
        )

    def test_read_npz_metadata_roundtrip(self):
        buffer = io.BytesIO()
        meta = {"analysis_id": "x", "hairstyle_id": 3, "feedback": "good"}
        np.savez_compressed(
            buffer,
            ground_truth=np.array([90.0], dtype=np.float32),
            metadata=np.array([json.dumps(meta)], dtype=str),
        )
        assert eng.read_npz_metadata(buffer.getvalue()) == meta
        assert eng.read_npz_metadata(b"not npz") is None

    def test_orphan_reactions_are_reported_not_counted(self):
        reactions = {("a", 1): (1.0, "synthesize"), ("old", 9): (3.0, "save")}
        result = eng.compute_engagement([_exp("a", 1)], reactions)
        assert result["summary"]["matched_pairs"] == 1
        assert result["summary"]["orphan_pairs"] == 1
        assert result["summary"]["orphan_by_kind"] == {"save": 1}
        assert result["summary"]["reaction_weight"] == 1.0


class TestSubjectCap:
    def test_same_subject_counts_once_per_cell(self):
        exposures = [_exp(f"a{i}", 1, subject="same") for i in range(50)]
        reactions = {(f"a{i}", 1): (3.0, "share") for i in range(50)}
        result = eng.compute_engagement(exposures, reactions)
        cell = result["groups"][G]["1"]
        assert cell["exposures"] == 1
        assert cell["reactions"] == 3.0
        assert cell["subjects"] == 1
        assert result["summary"]["capped_exposures"] == 49

    def test_subject_max_reaction_and_min_rank(self):
        exposures = [
            _exp("a1", 1, rank=3, subject="s"),
            _exp("a2", 1, rank=1, subject="s"),
            _exp("a3", 1, rank=2, subject="t"),
        ]
        reactions = {("a1", 1): (1.0, "synthesize"), ("a2", 1): (3.0, "save")}
        units, _ = eng.collapse_units(exposures, reactions)
        by_subject = {u.subject: u for u in units}
        assert by_subject["s"].reaction == (3.0, "save")
        assert by_subject["s"].rank == 1
        assert by_subject["t"].reaction == (0.0, "")

    def test_subjects_count_only_save_share(self):
        exposures = [_exp(f"a{i}", 1, subject=f"s{i}") for i in range(4)]
        reactions = {
            ("a0", 1): (3.0, "save"),
            ("a1", 1): (3.0, "share"),
            ("a2", 1): (1.0, "synthesize"),
            ("a3", 1): (2.0, "explicit_good"),
        }
        cell = eng.compute_engagement(exposures, reactions)["groups"][G]["1"]
        assert cell["subjects"] == 2

    def test_subject_of_prefers_user_id_and_hashes(self):
        logged_in = eng.subject_of({"user_id": "u-1", "image_hash": "h1"})
        same_user = eng.subject_of({"user_id": "u-1", "image_hash": "h2"})
        anon = eng.subject_of({"user_id": "anonymous", "image_hash": "h1"})
        assert logged_in == same_user
        assert anon != logged_in
        assert anon == eng.subject_of({"image_hash": "h1"})
        assert "u-1" not in logged_in and "h1" not in anon
        assert eng.subject_of({"user_id": "anonymous"}) is None

    def test_social_proof_exposures_are_excluded(self):
        exposures = [
            _exp("a", 1, subject="s1"),
            _exp("b", 1, subject="s2", social_proof=True),
        ]
        reactions = {("a", 1): (1.0, "synthesize"), ("b", 1): (3.0, "save")}
        result = eng.compute_engagement(exposures, reactions)
        assert result["groups"][G]["1"]["exposures"] == 1
        assert result["groups"][G]["1"]["reactions"] == 1.0
        summary = result["summary"]
        assert summary["social_proof_exposures"] == 1
        assert summary["social_proof_reactions"] == 1
        assert summary["orphan_pairs"] == 0


class TestExposures:
    def _item(self, created, cards, gender="male", face="둥근형", aid="a1", **extra):
        item = {
            "analysis_id": aid,
            "created_at": created,
            "gender": gender,
            "face_shape": face,
            "recommended_styles": cards,
            "image_hash": f"hash-{aid}",
        }
        item.update(extra)
        return item

    def test_ml_cards_only_and_window(self):
        cards = [
            {"hairstyle_id": 1, "rank": 1, "source": "ml"},
            {"hairstyle_id": 1, "rank": 2, "source": "ml"},  # 같은 분석 중복
            {"hairstyle_id": 2, "rank": 3, "source": "ml"},
            {"style_name": "트렌드", "rank": 4, "source": "trending"},
        ]
        items = [
            self._item("2026-10-01T00:00:00+00:00", cards),
            self._item("2026-08-01T00:00:00+00:00", cards, aid="old"),
        ]
        exposures, stats = eng.collect_exposures(
            items,
            NOW - timedelta(days=60),
            NOW,
            datetime(2026, 9, 17, tzinfo=timezone.utc),
        )
        assert [(e.hairstyle_id, e.rank, e.group) for e in exposures] == [
            (1, 1, G),
            (2, 3, G),
        ]
        assert stats.trending_exposures == 1
        assert stats.skipped_out_of_window == 1

    def test_decimal_and_str_hairstyle_ids(self):
        cards = [
            {"hairstyle_id": Decimal("3"), "rank": Decimal("1"), "source": "ml"},
            {"hairstyle_id": "4", "rank": Decimal("2"), "source": "ml"},
            {"hairstyle_id": 3.0, "rank": 3, "source": "ml"},  # 3 과 같은 스타일
        ]
        exposures, _ = eng.collect_exposures(
            [self._item("2026-10-01T00:00:00+00:00", cards)],
            NOW - timedelta(days=60),
            NOW,
        )
        assert [(e.hairstyle_id, e.rank) for e in exposures] == [(3, 1), (4, 2)]

    def test_engagement_reason_card_is_flagged(self):
        cards = [
            {"hairstyle_id": 1, "rank": 1, "source": "ml", "engagement_reason": True},
            {"hairstyle_id": 2, "rank": 2, "source": "ml"},
        ]
        exposures, stats = eng.collect_exposures(
            [self._item("2026-10-01T00:00:00+00:00", cards)],
            NOW - timedelta(days=60),
            NOW,
        )
        assert [e.social_proof for e in exposures] == [True, False]
        assert stats.social_proof_exposures == 1

    def test_untrusted_face_shape_and_unknown_gender(self):
        cards = [{"hairstyle_id": 1, "rank": 1, "source": "ml"}]
        items = [
            self._item("2026-09-10T00:00:00+00:00", cards, aid="pre"),
            self._item("2026-09-18T00:00:00", cards, gender=None, aid="nogender"),
            self._item("2026-09-21T00:00:00+00:00", cards, gender="neutral", aid="n"),
        ]
        exposures, stats = eng.collect_exposures(
            items,
            NOW - timedelta(days=60),
            NOW,
            datetime(2026, 9, 17, tzinfo=timezone.utc),
        )
        groups = {e.analysis_id: e.group for e in exposures}
        assert groups == {
            "pre": None,
            "nogender": "unknown|둥근형",
            "n": "unknown|둥근형",
        }
        assert stats.face_shape_untrusted_analyses == 1
        assert stats.first_gender_at == "2026-09-10T00:00:00+00:00"

    def test_subject_from_user_or_image(self):
        cards = [{"hairstyle_id": 1, "rank": 1, "source": "ml"}]
        items = [
            self._item("2026-10-01T00:00:00+00:00", cards, aid="a", user_id="u"),
            self._item("2026-10-02T00:00:00+00:00", cards, aid="b", user_id="u"),
            self._item("2026-10-03T00:00:00+00:00", cards, aid="c", image_hash=None),
        ]
        exposures, stats = eng.collect_exposures(items, NOW - timedelta(days=60), NOW)
        subjects = [e.subject for e in exposures]
        assert subjects[0] == subjects[1] is not None
        assert subjects[2] is None
        assert stats.no_subject_analyses == 1

    def test_payload_has_no_identifiers(self):
        cards = [{"hairstyle_id": 1, "rank": 1, "source": "ml"}]
        items = [
            self._item(
                "2026-10-01T00:00:00+00:00",
                cards,
                aid="analysis-secret",
                user_id="user-secret",
                image_hash="hash-secret",
            )
        ]
        exposures, _ = eng.collect_exposures(items, NOW - timedelta(days=60), NOW)
        reactions = {("analysis-secret", 1): (3.0, "save")}
        payload = eng.build_payload(
            exposures,
            reactions,
            since=NOW - timedelta(days=60),
            until=NOW,
            weights=eng.DEFAULT_WEIGHTS,
            include_explicit=True,
            generated_at=NOW,
        )
        text = json.dumps(payload, ensure_ascii=False)
        for secret in (
            "analysis-secret",
            "user-secret",
            "hash-secret",
            exposures[0].subject,
        ):
            assert secret not in text
        assert payload["version"] == "eng_20261102T000000Z"
        table = EngagementTable.from_payload(json.loads(text))
        assert table.version == payload["version"]
        assert table.groups[G][1].subjects == 1


# ========== 공격 회귀 ==========


class TestAttackRegression:
    def _base(self):
        """정상 사용자 300명이 10개 스타일을 고르게 본 세계"""
        import random

        rng = random.Random(3)
        exposures, reactions = [], {}
        for i in range(300):
            aid = f"n{i}"
            for rank, hid in enumerate(rng.sample(range(10), 3), 1):
                exposures.append(_exp(aid, hid, rank, subject=f"user{i}"))
                if rng.random() < 0.3:
                    reactions[(aid, hid)] = (1.0, "synthesize")
        return exposures, reactions

    def _bonus(self, exposures, reactions, hid=7):
        payload = eng.build_payload(
            exposures,
            reactions,
            since=NOW - timedelta(days=60),
            until=NOW,
            weights=eng.DEFAULT_WEIGHTS,
            include_explicit=True,
        )
        table = EngagementTable.from_payload(payload)
        return compute_bonus(table.lift(G, hid), 3.0, 1.5), table

    def test_repeated_share_with_same_image_cannot_move_cell(self):
        exposures, reactions = self._base()
        before, _ = self._bonus(exposures, reactions)
        # 같은 사진으로 분석 200번 + 매번 7번 스타일 share
        for i in range(200):
            aid = f"x{i}"
            exposures.append(_exp(aid, 7, 1, subject="attacker-image"))
            reactions[(aid, 7)] = (3.0, "share")
        after, table = self._bonus(exposures, reactions)
        assert after - before < 0.3
        assert table.groups[G][7].subjects <= 1

    def test_save_without_synthesis_marker_is_not_recorded(self, monkeypatch):
        """이벤트 API 만 두드린 save 는 npz 가 생기지 않아 집계에 들어올 수 없다"""
        from botocore.exceptions import ClientError

        from services.mlops import implicit_signals as isig
        from services.mlops.feedback_features import (
            FACE_FEATURE_FIELDS,
            SKIN_FEATURE_FIELDS,
        )

        monkeypatch.setenv("MLOPS_ENABLED", "true")
        store = MagicMock(enabled=True)
        store.s3_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadObject"
        )
        analysis = {
            "analysis_id": "11111111-2222-3333-4444-555555555555",
            "recommended_styles": [{"hairstyle_id": 7, "source": "ml", "rank": 1}],
            **{f: 1.0 for f in FACE_FEATURE_FIELDS + SKIN_FEATURE_FIELDS},
        }
        result = isig.record_style_signal(
            analysis["analysis_id"],
            "save",
            hairstyle_id=7,
            result_id="abcdef0123456789",
            store=store,
            analysis_getter=lambda _id: analysis,
        )
        assert result["reason"] == "no_synthesis_marker"
        store.s3_client.put_object.assert_not_called()


# ========== 서빙: 테이블 / 보너스 ==========


class TestTable:
    def test_group_falls_back_to_style_then_zero(self):
        table = EngagementTable.from_payload(BOOST_PAYLOAD)
        assert table.lift(G, 105) == pytest.approx(1.0)
        assert table.lift("female|둥근형", 105) == pytest.approx(0.2)  # 스타일
        assert table.lift(G, 999) == 0.0

    @pytest.mark.parametrize(
        "payload, rejected",
        [
            (None, False),
            ([], False),
            ({"schema_version": 99, "version": "v", "mu": 0.5}, False),
            ({"schema_version": 1, "mu": 0.5}, False),
            ({"schema_version": 1, "version": "v", "mu": "x"}, False),
            (_payload(styles={"1": {"rate": -1, "exposures": 1}}), False),
            (_payload(styles={"abc": _entry(0.5)}), False),
            (_payload(mu=0.0), True),
            (_payload(mu=-0.1), True),
            (_payload(styles={}), True),
        ],
    )
    def test_invalid_payload_raises(self, payload, rejected):
        with pytest.raises((ValueError, TypeError)) as info:
            EngagementTable.from_payload(payload)
        assert isinstance(info.value, TableRejected) is rejected

    def test_bonus_cap(self):
        assert compute_bonus(10.0, 3.0, 3.0) == 3.0
        assert compute_bonus(-10.0, 3.0, 3.0) == -3.0
        assert compute_bonus(0.5, 3.0, 3.0) == pytest.approx(1.5)
        assert compute_bonus(1.0, 3.0, -2.0) == 2.0  # 음수 상한도 절댓값으로

    def test_default_cap_is_half_margin(self):
        from config.settings import Settings

        fields = Settings.model_fields
        assert fields["REC_ENGAGEMENT_CAP"].default == pytest.approx(
            fields["REC_EXPLORE_MARGIN"].default / 2
        )

    def test_plan_bonus(self, base_settings, monkeypatch):
        monkeypatch.setattr(settings, "REC_ENGAGEMENT_ENABLED", True)
        store = _FakeStore(EngagementTable.from_payload(BOOST_PAYLOAD))
        plan = rec_engagement.build_plan(
            image_hash=IMAGE_HASH,
            user_id=None,
            gender="male",
            face_shape="둥근형",
            store=store,
        )
        assert plan.arm == ARM_TREATMENT
        assert plan.bonus(105) == 3.0
        assert plan.bonus(100) == pytest.approx(-1.8)
        assert plan.bonus(Decimal("105")) == 3.0
        assert plan.bonus(None) == 0.0
        assert plan.bonus(12345) == 0.0


class TestStoreCache:
    def _loader(self, bodies):
        calls = []

        def load():
            calls.append(1)
            body = bodies[min(len(calls), len(bodies)) - 1]
            if isinstance(body, Exception):
                raise body
            return body

        return load, calls

    @staticmethod
    def _body(**kw):
        return json.dumps(_payload(**kw)).encode()

    def test_ttl_cache(self):
        clock = [0.0]
        load, calls = self._loader([self._body(), self._body(version="v2")])
        store = EngagementStore(load, ttl_seconds=3600, clock=lambda: clock[0])

        assert store.get().version == "eng_test_v1"
        clock[0] = 3599
        assert store.get().version == "eng_test_v1"
        assert len(calls) == 1
        clock[0] = 3600
        assert store.get().version == "v2"
        assert len(calls) == 2

    def test_failure_is_cached_and_retried(self):
        clock = [0.0]
        load, calls = self._loader([RuntimeError("s3 down"), self._body()])
        store = EngagementStore(
            load, ttl_seconds=3600, retry_seconds=300, clock=lambda: clock[0]
        )
        assert store.get() is None
        clock[0] = 299
        assert store.get() is None
        assert len(calls) == 1  # 요청마다 S3 를 부르지 않는다
        clock[0] = 300
        assert store.get().version == "eng_test_v1"

    def test_stale_while_error_keeps_last_good(self):
        clock = [0.0]
        load, calls = self._loader([self._body(), RuntimeError("s3 down")])
        store = EngagementStore(
            load, ttl_seconds=3600, retry_seconds=300, clock=lambda: clock[0]
        )
        assert store.get().version == "eng_test_v1"
        clock[0] = 3600
        with patch.object(rec_engagement, "log_structured") as log:
            assert store.get().version == "eng_test_v1"  # 갱신 실패 - 마지막 정상
        assert log.call_args.args[0] == "engagement_table_rejected"
        assert log.call_args.args[1]["reason"] == "load_error"
        clock[0] = 3600 + 299
        assert store.get().version == "eng_test_v1"
        assert len(calls) == 2  # 실패 후 재시도 간격 동안 다시 읽지 않는다

    def test_last_good_dropped_after_max_age(self):
        clock = [0.0]
        wall = [datetime(2026, 11, 2, tzinfo=timezone.utc)]
        body = self._body(generated_at=wall[0])
        load, _ = self._loader([body, RuntimeError("s3 down")])
        store = EngagementStore(
            load,
            ttl_seconds=3600,
            clock=lambda: clock[0],
            max_age_days=3,
            wall_clock=lambda: wall[0],
        )
        assert store.get() is not None
        clock[0] = 3600
        wall[0] += timedelta(days=3, minutes=1)
        with patch.object(rec_engagement, "log_structured") as log:
            assert store.get() is None
        reasons = [c.args[1]["reason"] for c in log.call_args_list]
        assert reasons == ["load_error", "stale"]

    def test_cached_table_becomes_stale(self):
        wall = [datetime(2026, 11, 2, tzinfo=timezone.utc)]
        store = EngagementStore(
            lambda: self._body(generated_at=wall[0] - timedelta(days=2, hours=23)),
            ttl_seconds=3600,
            clock=lambda: 0.0,
            max_age_days=3,
            wall_clock=lambda: wall[0],
        )
        assert store.get() is not None
        wall[0] += timedelta(hours=2)
        assert store.get() is None

    @pytest.mark.parametrize(
        "body, reason",
        [
            (b"{not json", "parse_error"),
            (b"[]", "parse_error"),
            (json.dumps({"schema_version": 1}).encode(), "parse_error"),
            (json.dumps(_payload(mu=0.0)).encode(), "mu_not_positive"),
            (json.dumps(_payload(styles={})).encode(), "empty_styles"),
            (
                json.dumps(
                    _payload(
                        generated_at=datetime.now(timezone.utc) - timedelta(days=4)
                    )
                ).encode(),
                "stale",
            ),
            (
                json.dumps(
                    {k: v for k, v in _payload().items() if k != "generated_at"}
                ).encode(),
                "stale",
            ),
        ],
    )
    def test_rejected_tables_give_none_and_log(self, body, reason):
        store = EngagementStore(lambda: body, ttl_seconds=60, max_age_days=3)
        with patch.object(rec_engagement, "log_structured") as log:
            assert store.get() is None
        event, fields = log.call_args.args
        assert event == "engagement_table_rejected"
        assert fields["reason"] == reason

    def test_default_ttl_and_age_from_settings(self, monkeypatch):
        monkeypatch.setattr(settings, "REC_ENGAGEMENT_CACHE_TTL_SECONDS", 120)
        monkeypatch.setattr(settings, "REC_ENGAGEMENT_MAX_AGE_DAYS", 2)
        store = EngagementStore(lambda: b"")
        assert store.ttl_seconds == 120.0
        assert store.max_age == timedelta(days=2)

    def test_s3_failure_means_no_bonus(self, base_settings, monkeypatch):
        monkeypatch.setattr(settings, "REC_ENGAGEMENT_ENABLED", True)

        def boom():
            raise RuntimeError("NoSuchKey")

        rec_engagement.reset_engagement_store(EngagementStore(boom, ttl_seconds=60))
        out = {}
        result = _recommend(_service(), out=out)
        assert out == {}
        assert all("engagement_bonus" not in c for c in _ml_cards(result))
        assert "engagement_arm" not in result["meta"]


# ========== 기존 응답과 동일 (플래그 꺼짐 / 대조군 / 테이블 없음) ==========


def _snapshot():
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def _dump(result):
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


EXPLORE_CASES = {
    "explore_off": (False, 0.5),
    "explore_arm": (True, 1.0),
    "explore_control": (True, 0.0),
}


class TestByteIdentical:
    @pytest.mark.parametrize("case", list(EXPLORE_CASES))
    def test_flag_off_matches_snapshot(self, base_settings, monkeypatch, case):
        explore, ratio = EXPLORE_CASES[case]
        monkeypatch.setattr(settings, "REC_EXPLORE_ENABLED", explore)
        monkeypatch.setattr(settings, "REC_EXPLORE_RATIO", ratio)
        store = _install_table(BOOST_PAYLOAD)

        result = _recommend(_service())

        assert _dump(result) == _dump(_snapshot()[case])
        assert store.calls == 0  # 꺼져 있으면 S3 테이블도 안 본다

    @pytest.mark.parametrize("case", list(EXPLORE_CASES))
    def test_control_arm_matches_snapshot(self, base_settings, monkeypatch, case):
        explore, ratio = EXPLORE_CASES[case]
        monkeypatch.setattr(settings, "REC_EXPLORE_ENABLED", explore)
        monkeypatch.setattr(settings, "REC_EXPLORE_RATIO", ratio)
        monkeypatch.setattr(settings, "REC_ENGAGEMENT_ENABLED", True)
        monkeypatch.setattr(settings, "REC_ENGAGEMENT_RATIO", 0.0)  # 전원 대조군
        _install_table(BOOST_PAYLOAD)
        out = {}
        service = _service()

        result = _recommend(service, out=out)

        assert _dump(result) == _dump(_snapshot()[case])
        assert out == {
            "engagement_arm": ARM_CONTROL,
            "engagement_version": "eng_test_v1",
            "engagement_applied": False,
        }
        if not explore:
            assert (
                service.ml_recommender.recommend_top_k.call_args_list[0].kwargs["k"]
                == 3
            )

    @pytest.mark.parametrize("case", list(EXPLORE_CASES))
    def test_missing_table_matches_snapshot(self, base_settings, monkeypatch, case):
        explore, ratio = EXPLORE_CASES[case]
        monkeypatch.setattr(settings, "REC_EXPLORE_ENABLED", explore)
        monkeypatch.setattr(settings, "REC_EXPLORE_RATIO", ratio)
        monkeypatch.setattr(settings, "REC_ENGAGEMENT_ENABLED", True)
        _install_table(None)
        out = {}
        result = _recommend(_service(), out=out)
        assert _dump(result) == _dump(_snapshot()[case])
        assert out == {}

    def test_no_image_hash_is_outside_experiment(self, base_settings, monkeypatch):
        monkeypatch.setattr(settings, "REC_ENGAGEMENT_ENABLED", True)
        store = _install_table(BOOST_PAYLOAD)
        out = {}
        service = _service()
        result = _recommend(service, image_hash=None, out=out)
        assert out == {}
        assert store.calls == 0
        assert service.ml_recommender.recommend_top_k.call_args_list[0].kwargs["k"] == 3
        assert all("engagement_bonus" not in c for c in _ml_cards(result))


# ========== 4가지 플래그 조합 ==========


class TestFlagCombinations:
    def test_explore_off_engagement_off_uses_k3(self, base_settings):
        service = _service()
        result = _recommend(service)
        assert service.ml_recommender.recommend_top_k.call_args_list[0].kwargs["k"] == 3
        assert [c["hairstyle_id"] for c in _ml_cards(result)] == [100, 101, 102]

    def test_explore_on_engagement_off_is_existing_explore(
        self, base_settings, monkeypatch
    ):
        monkeypatch.setattr(settings, "REC_EXPLORE_ENABLED", True)
        monkeypatch.setattr(settings, "REC_EXPLORE_RATIO", 1.0)
        service = _service()
        result = _recommend(service)
        assert (
            service.ml_recommender.recommend_top_k.call_args_list[0].kwargs["k"] == 20
        )
        assert result["meta"]["rec_policy"] == POLICY_EXPLORE
        assert "engagement_arm" not in result["meta"]

    def test_explore_off_engagement_treatment_uses_pool_and_bonus(
        self, base_settings, monkeypatch
    ):
        monkeypatch.setattr(settings, "REC_ENGAGEMENT_ENABLED", True)
        _install_table(BOOST_PAYLOAD)
        out = {}
        service = _service()

        result = _recommend(service, out=out)

        first_call = service.ml_recommender.recommend_top_k.call_args_list[0]
        assert first_call.kwargs["k"] == 20
        assert first_call.kwargs["log_selection"] is False
        cards = _ml_cards(result)
        # 105: 60.8 + 3.0 = 63.8 -> 1위, 100: 62.4 - 1.8 = 60.6 -> 탈락
        assert [c["hairstyle_id"] for c in cards] == [105, 101, 102]
        assert [c["rank"] for c in cards] == [1, 2, 3]
        assert [c["engagement_bonus"] for c in cards] == [3.0, 0.0, 0.0]
        assert [c["raw_score"] for c in cards] == [60.8, 62.1, 61.9]  # 보너스 전
        assert [c["score"] for c in cards] == [0.64, 0.62, 0.62]  # 보너스 포함
        assert [c["ml_rank"] for c in cards] == [6, 2, 3]  # 모델 점수 기준 풀 순위
        assert [c["propensity"] for c in cards] == [1.0, 1.0, 1.0]
        meta = result["meta"]
        assert meta["rec_policy"] == POLICY_DETERMINISTIC
        assert meta["experiment_arm"] is None
        assert meta["tie_band_size"] is None
        assert meta["engagement_arm"] == ARM_TREATMENT
        assert meta["engagement_version"] == "eng_test_v1"
        assert "engagement_applied" not in meta  # 응답이 아니라 저장·로그용
        # 계측용 top_score 는 모델 점수
        assert meta["top_score"] == 62.1
        assert out == {
            "engagement_arm": ARM_TREATMENT,
            "engagement_version": "eng_test_v1",
            "engagement_applied": True,
        }

    def test_explore_on_engagement_treatment_bonus_before_tie_band(
        self, base_settings, monkeypatch
    ):
        monkeypatch.setattr(settings, "REC_EXPLORE_ENABLED", True)
        monkeypatch.setattr(settings, "REC_EXPLORE_RATIO", 1.0)
        monkeypatch.setattr(settings, "REC_ENGAGEMENT_ENABLED", True)
        _install_table(BOOST_PAYLOAD)

        picks = set()
        for i in range(60):
            result = _recommend(_service(), image_hash=f"{i:064x}")
            cards = _ml_cards(result)
            meta = result["meta"]
            assert meta["rec_policy"] == POLICY_EXPLORE
            assert meta["experiment_arm"] == ARM_EXPLORE
            assert meta["engagement_arm"] == ARM_TREATMENT
            # 동점권은 보너스 포함 점수 기준: 63.8 - 3 = 60.8 이상 -> 105, 101~104
            assert meta["tie_band_size"] == 5
            adjusted = [c["raw_score"] + c["engagement_bonus"] for c in cards]
            assert all(a >= 63.8 - 3.0 - 1e-6 for a in adjusted)
            assert adjusted == sorted(adjusted, reverse=True)
            assert 100 not in [c["hairstyle_id"] for c in cards]  # 감점으로 동점권 밖
            assert all(0.0 < c["propensity"] <= 1.0 for c in cards)
            picks.update(c["hairstyle_id"] for c in cards)
        assert picks == {101, 102, 103, 104, 105}

    def test_explore_control_engagement_treatment_is_deterministic_adjusted(
        self, base_settings, monkeypatch
    ):
        monkeypatch.setattr(settings, "REC_EXPLORE_ENABLED", True)
        monkeypatch.setattr(settings, "REC_EXPLORE_RATIO", 0.0)
        monkeypatch.setattr(settings, "REC_ENGAGEMENT_ENABLED", True)
        _install_table(BOOST_PAYLOAD)
        result = _recommend(_service())
        assert [c["hairstyle_id"] for c in _ml_cards(result)] == [105, 101, 102]
        meta = result["meta"]
        assert meta["experiment_arm"] == EXPLORE_CONTROL
        assert meta["rec_policy"] == POLICY_DETERMINISTIC
        assert meta["tie_band_size"] == 5

    def test_independent_assignment(self):
        both = {"tt": 0, "tc": 0, "ct": 0, "cc": 0}
        from services import rec_exploration

        for i in range(4000):
            h = f"{i:064x}"
            e = rec_exploration.assign_arm(None, h, 0.5) == ARM_EXPLORE
            g = rec_engagement.assign_arm(None, h, 0.5) == ARM_TREATMENT
            both[("t" if e else "c") + ("t" if g else "c")] += 1
        for count in both.values():
            assert 0.2 < count / 4000 < 0.3

    def test_full_ranking_uses_adjusted_scores(self, base_settings, monkeypatch):
        monkeypatch.setattr(settings, "REC_ENGAGEMENT_ENABLED", True)
        _install_table(BOOST_PAYLOAD)
        result = _recommend(_service())
        ranking = result["full_ranking"]
        shown = [c["style_name"] for c in _ml_cards(result)]
        assert [r["style_name"] for r in ranking[:3]] == shown
        assert ranking[0] == {"style_name": "스타일05", "score": 0.64}
        names = [r["style_name"] for r in ranking]
        assert names.index("스타일00") > names.index("스타일05")
        assert {"style_name": "스타일00", "score": 0.61} in ranking
        assert len(ranking) == 20
        scores = [r["score"] for r in ranking[3:]]
        assert scores == sorted(scores, reverse=True)

    def test_full_ranking_bonus_only_within_pool(self, base_settings, monkeypatch):
        """풀(상위 POOL) 밖 후보는 화면에 나올 수 없으므로 전체 랭킹에서도 보너스 없음"""
        monkeypatch.setattr(settings, "REC_ENGAGEMENT_ENABLED", True)
        monkeypatch.setattr(settings, "REC_EXPLORE_POOL", 5)
        _install_table(
            _payload(
                mu=0.5,
                styles={"110": _entry(0.5)},
                groups={G: {"110": _entry(1.0, 200), "101": _entry(1.0, 200)}},
            )
        )
        result = _recommend(_service())
        by_name = {r["style_name"]: r["score"] for r in result["full_ranking"]}
        assert by_name["스타일10"] == 0.58  # 58.2 그대로 (풀 밖)
        assert by_name["스타일01"] == 0.65  # 62.1 + 3.0
        assert "스타일10" not in [c["style_name"] for c in _ml_cards(result)]

    def test_pool_failure_records_assignment_not_applied(
        self, base_settings, monkeypatch
    ):
        monkeypatch.setattr(settings, "REC_ENGAGEMENT_ENABLED", True)
        monkeypatch.setattr(settings, "FULL_RANKING_LIMIT", 0)
        _install_table(BOOST_PAYLOAD)
        ranked = _ranked()
        service = _service(ranked)
        service.ml_recommender.recommend_top_k.side_effect = [
            RuntimeError("pool boom"),
            [dict(r) for r in ranked[:3]],
        ]
        out = {}
        result = _recommend(service, out=out)
        assert [c["hairstyle_id"] for c in _ml_cards(result)] == [100, 101, 102]
        assert all("engagement_bonus" not in c for c in _ml_cards(result))
        assert "engagement_arm" not in result["meta"]
        # 배정 기준 분석(ITT): 실험군 배정은 남기고 미적용 표시
        assert out == {
            "engagement_arm": ARM_TREATMENT,
            "engagement_version": "eng_test_v1",
            "engagement_applied": False,
        }


# ========== 사유 문구 ==========


class TestReason:
    def _plan(
        self, payload, gender="male", face_shape="둥근형", seed=IMAGE_HASH, subjects=0
    ):
        return rec_engagement.EngagementPlan(
            arm=ARM_TREATMENT,
            table=EngagementTable.from_payload(payload),
            gender=rec_engagement.normalize_gender(gender),
            face_shape=face_shape,
            seed_key=seed,
            beta=3.0,
            cap=3.0,
            reason_min_lift=0.3,
            reason_min_exposures=50,
            reason_min_subjects=subjects,
        )

    def test_reason_when_lift_exposure_and_subjects_enough(self):
        plan = self._plan(BOOST_PAYLOAD, subjects=15)
        reason = plan.reason(105)
        assert reason is not None
        assert "둥근형 남성" in reason
        assert reason in [
            t.format(who="둥근형 남성") for t in rec_engagement.REASON_TEMPLATES
        ]

    def test_save_share_subjects_required(self):
        """합성·좋아요로 생긴 lift 만으로는 "저장한" 문구가 나가지 않는다"""
        payload = _payload(
            groups={
                G: {
                    "1": _entry(1.0, 100, 40.0, 14),
                    "2": _entry(1.0, 100, 40.0, 15),
                }
            }
        )
        plan = self._plan(payload, subjects=15)
        assert plan.reason(1) is None
        assert plan.reason(2) is not None

    def test_unknown_gender_uses_face_shape_only(self):
        payload = _payload(groups={"unknown|둥근형": {"105": _entry(1.0, 80)}})
        reason = self._plan(payload, gender="neutral").reason(105)
        assert reason is not None
        assert "둥근형 얼굴" in reason
        assert "남성" not in reason and "여성" not in reason

    def test_no_reason_below_thresholds(self):
        payload = _payload(
            groups={
                G: {
                    "1": _entry(0.64, 500),  # lift 0.28 < 0.3
                    "2": _entry(1.0, 49),  # 노출 부족
                    "3": _entry(0.66, 50),  # lift 0.32, 노출 50 -> 문구
                }
            },
            styles={"4": _entry(1.0, 500)},  # 셀 없음 -> 문구 없음
        )
        plan = self._plan(payload)
        assert plan.reason(1) is None
        assert plan.reason(2) is None
        assert plan.reason(3) is not None
        assert plan.reason(4) is None

    def test_lift_is_against_group_mean(self):
        payload = _payload(
            groups={G: {"1": _entry(0.9, 100)}},
            group_totals={G: {"mean_rate": 0.8}},
        )
        assert self._plan(payload).reason(1) is None  # 0.9 / 0.8 = +12.5%

    def test_template_choice_is_deterministic_and_varies(self):
        payload = _payload(groups={"female|긴형": {"7": _entry(2.0, 200)}})
        first = self._plan(payload, gender="female", face_shape="긴형").reason(7)
        again = self._plan(payload, gender="female", face_shape="긴형").reason(7)
        assert first == again
        variants = {
            self._plan(
                payload, gender="female", face_shape="긴형", seed=f"{i:064x}"
            ).reason(7)
            for i in range(40)
        }
        assert len(variants) == len(rec_engagement.REASON_TEMPLATES)

    def test_reason_replaced_only_in_treatment_and_flagged(
        self, base_settings, monkeypatch
    ):
        monkeypatch.setattr(settings, "REC_ENGAGEMENT_ENABLED", True)
        _install_table(BOOST_PAYLOAD)
        cards = _ml_cards(_recommend(_service()))
        assert "둥근형 남성" in cards[0]["reason"]  # 105
        assert cards[0]["engagement_reason"] is True
        assert all(c["reason"].startswith("ML 모델 추천") for c in cards[1:])
        assert all("engagement_reason" not in c for c in cards[1:])

        monkeypatch.setattr(settings, "REC_ENGAGEMENT_RATIO", 0.0)
        control = _ml_cards(_recommend(_service()))
        assert all("사용자" not in c["reason"] for c in control)
        assert all("engagement_reason" not in c for c in control)


# ========== 엔드포인트 ==========


def _post(service):
    from PIL import Image
    from fastapi.testclient import TestClient

    from core.dependencies import get_face_detection_service, get_hybrid_service
    from main import app

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color="white").save(buffer, format="JPEG")
    features = MagicMock(
        face_shape="둥근형",
        skin_tone="봄웜",
        face_features=[1.4, 120.0, 130.0, 110.0, 0.92, 0.85],
        skin_features=[45.0, 15.0],
        confidence=0.9,
    )
    detector = MagicMock()
    detector.detect_face.return_value = {"has_face": True, "features": features}

    app.dependency_overrides[get_face_detection_service] = lambda: detector
    app.dependency_overrides[get_hybrid_service] = lambda: service
    try:
        with patch(
            "services.trending_style_service.get_trending_style_service"
        ) as trending, patch(
            "api.endpoints.analyze.save_to_database", return_value="analysis-1"
        ) as save, patch(
            "api.endpoints.analyze.log_structured"
        ) as log, patch(
            "api.endpoints.analyze.limiter.enabled", False
        ), patch(
            "api.endpoints.analyze.get_style_image_service"
        ) as images, TestClient(
            app
        ) as client:
            trending.return_value.pick_trending.return_value = [
                dict(t) for t in TRENDING
            ]
            images.return_value.build_image_url.return_value = None
            response = client.post(
                "/api/v2/analyze-hybrid",
                files={"file": ("face.jpg", buffer.getvalue(), "image/jpeg")},
                data={"gender": "male"},
            )
    finally:
        app.dependency_overrides.pop(get_face_detection_service, None)
        app.dependency_overrides.pop(get_hybrid_service, None)
    return response, save, log


def _analysis_complete(log):
    events = [c.args[1] for c in log.call_args_list if c.args[0] == "analysis_complete"]
    assert len(events) == 1
    return events[0]


def _response_body(response):
    body = response.json()
    body.pop("processing_time", None)
    return _dump(body)


class TestEndpoint:
    def test_control_response_identical_to_flag_off(self, base_settings, monkeypatch):
        off_response, off_save, off_log = _post(_service())

        monkeypatch.setattr(settings, "REC_ENGAGEMENT_ENABLED", True)
        monkeypatch.setattr(settings, "REC_ENGAGEMENT_RATIO", 0.0)
        _install_table(BOOST_PAYLOAD)
        response, save, log = _post(_service())

        assert response.status_code == 200, response.text
        assert _response_body(response) == _response_body(off_response)
        saved = save.call_args.kwargs["analysis_result"]
        assert saved["engagement_arm"] == ARM_CONTROL
        assert saved["engagement_version"] == "eng_test_v1"
        assert saved["engagement_applied"] is False
        assert "experiment_id" not in saved  # 탐색 실험 필드와 섞이지 않는다
        event = _analysis_complete(log)
        assert event["engagement_arm"] == ARM_CONTROL
        assert event["engagement_version"] == "eng_test_v1"
        assert event["engagement_applied"] is False

        off_saved = off_save.call_args.kwargs["analysis_result"]
        assert "engagement_arm" not in off_saved
        assert "engagement_applied" not in off_saved
        off_event = _analysis_complete(off_log)
        assert off_event["engagement_arm"] is None
        assert off_event["engagement_version"] is None
        assert off_event["engagement_applied"] is None

    def test_treatment_fields_in_response_db_and_log(self, base_settings, monkeypatch):
        monkeypatch.setattr(settings, "REC_EXPLORE_ENABLED", True)
        monkeypatch.setattr(settings, "REC_EXPLORE_RATIO", 1.0)
        monkeypatch.setattr(settings, "REC_ENGAGEMENT_ENABLED", True)
        _install_table(BOOST_PAYLOAD)
        response, save, log = _post(_service())

        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["meta"]["engagement_arm"] == ARM_TREATMENT
        assert all("engagement_bonus" in c for c in _ml_cards(data))
        saved = save.call_args.kwargs["analysis_result"]
        assert saved["engagement_arm"] == ARM_TREATMENT
        assert saved["engagement_applied"] is True
        assert saved["experiment_id"] == "rec_explore_v1"
        assert saved["ab_variant"] == ARM_EXPLORE
        saved_cards = _ml_cards({"recommendations": saved["recommendations"]})
        assert all("engagement_bonus" in c for c in saved_cards)
        # 사유가 바뀐 카드 표식도 저장된다 (집계에서 제외용)
        flagged = [c for c in saved_cards if c.get("engagement_reason")]
        assert [c["hairstyle_id"] for c in flagged] == [
            c["hairstyle_id"] for c in _ml_cards(data) if c.get("engagement_reason")
        ]
        event = _analysis_complete(log)
        assert event["engagement_arm"] == ARM_TREATMENT
        assert event["engagement_applied"] is True
        assert event["experiment_arm"] == ARM_EXPLORE

    def test_repository_and_connection_store_fields(self):
        from database.dynamodb_repository import DynamoDBAnalysisRepository

        with patch(
            "database.dynamodb_repository.save_analysis", return_value="id-1"
        ) as save:
            DynamoDBAnalysisRepository().save_analysis(
                image_hash="h",
                analysis_result={
                    "analysis": {"face_shape": "둥근형", "personal_color": "봄웜"},
                    "recommendations": [],
                    "engagement_arm": ARM_TREATMENT,
                    "engagement_version": "eng_x",
                    "engagement_applied": False,
                },
                processing_time=0.1,
                detection_method="ml",
            )
        data = save.call_args.args[0]
        assert data["engagement_arm"] == ARM_TREATMENT
        assert data["engagement_version"] == "eng_x"
        assert data["engagement_applied"] is False

        import database.dynamodb_connection as conn

        table = MagicMock()
        with patch.object(conn, "dynamodb_table", table), patch.object(
            conn, "dynamodb_enabled", True
        ):
            conn.save_analysis(
                {
                    "face_shape": "둥근형",
                    "engagement_arm": ARM_CONTROL,
                    "engagement_version": "eng_x",
                    "engagement_applied": False,
                }
            )
            conn.save_analysis({"face_shape": "둥근형"})
        first = table.put_item.call_args_list[0].kwargs["Item"]
        second = table.put_item.call_args_list[1].kwargs["Item"]
        assert first["engagement_arm"] == ARM_CONTROL
        assert first["engagement_version"] == "eng_x"
        assert first["engagement_applied"] is False
        for field in ("engagement_arm", "engagement_version", "engagement_applied"):
            assert field not in second


# ========== 집계 스크립트 ==========


class TestScript:
    def test_parse_weights(self):
        import argparse

        from scripts.build_style_engagement import parse_weights

        weights = parse_weights("save=5, explicit_good=0")
        assert weights["save"] == 5.0
        assert weights["explicit_good"] == 0.0
        assert weights["share"] == 3.0
        with pytest.raises(argparse.ArgumentTypeError):
            parse_weights("bogus=1")

    def test_load_samples_and_list_keys_with_fake_s3(self):
        from scripts.build_style_engagement import list_feedback_keys, load_samples

        def npz(meta):
            buffer = io.BytesIO()
            np.savez_compressed(
                buffer, metadata=np.array([json.dumps(meta)], dtype=str)
            )
            return buffer.getvalue()

        objects = {
            "feedback/pending/implicit/a_1.npz": npz(
                {"analysis_id": "a", "hairstyle_id": 1, "source": "implicit",
                 "signal_type": "save"}
            ),
            "feedback/processed/batch_x/a_1.npz": npz(
                {"analysis_id": "a", "hairstyle_id": 1, "source": "implicit",
                 "signal_type": "synthesize"}
            ),
            "feedback/processed/batch_x/old.npz": npz(
                {"analysis_id": "b", "hairstyle_id": 3, "feedback": "good"}
            ),
            "feedback/pending/broken.npz": b"xx",
        }  # fmt: skip
        s3 = MagicMock()
        s3.get_paginator.return_value.paginate.side_effect = lambda Bucket, Prefix: [
            {
                "Contents": [{"Key": k} for k in objects if k.startswith(Prefix)]
                + [{"Key": f"{Prefix}metadata.json"}]
            }
        ]
        s3.get_object.side_effect = lambda Bucket, Key: {
            "Body": io.BytesIO(objects[Key])
        }

        keys = list_feedback_keys(s3, "bucket")
        assert sorted(keys) == sorted(objects)
        samples, stats = load_samples(s3, "bucket", keys, max_workers=2)
        assert sorted((s.analysis_id, s.hairstyle_id, s.kind) for s in samples) == [
            ("a", 1, "save"),
            ("a", 1, "synthesize"),
            ("b", 3, "explicit_good"),
        ]
        assert stats["no_metadata"] == 1
        # pending + processed 같은 쌍은 하나로
        assert eng.reduce_reactions(samples, eng.DEFAULT_WEIGHTS)[("a", 1)] == (
            3.0,
            "save",
        )

    def test_scan_is_read_only_with_projection(self):
        from scripts.build_style_engagement import scan_analyses

        table = MagicMock()
        table.scan.side_effect = [
            {"Items": [{"analysis_id": "a"}], "LastEvaluatedKey": {"analysis_id": "a"}},
            {"Items": [{"analysis_id": "b"}]},
        ]
        items = list(scan_analyses(table, NOW))
        assert [i["analysis_id"] for i in items] == ["a", "b"]
        first = table.scan.call_args_list[0].kwargs
        projected = set(first["ExpressionAttributeNames"].values())
        assert {"image_hash", "user_id"} <= projected  # 주체당 상한용
        assert "device_id" not in projected
        assert first["ExpressionAttributeValues"] == {":since": "2026-11-02"}
        assert table.scan.call_args_list[1].kwargs["ExclusiveStartKey"] == {
            "analysis_id": "a"
        }
        assert not table.put_item.called

    def test_write_guard(self):
        from scripts.build_style_engagement import write_guard

        good = {"summary": {"exposures": 1000}, "mu": 0.4, "styles": {"1": {}}}
        kw = dict(min_exposures=500, max_read_error_ratio=0.05)
        assert write_guard(good, read_errors=5, keys_total=100, **kw) == []
        assert len(write_guard(good, read_errors=6, keys_total=100, **kw)) == 1
        assert write_guard(
            {"summary": {"exposures": 10}, "mu": 0.0, "styles": {}},
            read_errors=0,
            keys_total=0,
            **kw,
        ) == ["노출 10 < 500", "mu 0.0 <= 0", "styles 비어 있음"]

    def _run_main(self, monkeypatch, tmp_path, *, read_error=False, items=None):
        import sys
        import types

        from scripts import build_style_engagement as script

        cards = [{"hairstyle_id": 1, "rank": 1, "source": "ml"}]
        items = items or [
            {
                "analysis_id": f"a{i}",
                "created_at": (NOW - timedelta(days=1)).isoformat(),
                "gender": "male",
                "face_shape": "둥근형",
                "recommended_styles": cards,
                "image_hash": f"h{i}",
            }
            for i in range(20)
        ]
        meta = {"analysis_id": "a0", "hairstyle_id": 1, "feedback": "good"}
        buffer = io.BytesIO()
        np.savez_compressed(buffer, metadata=np.array([json.dumps(meta)], dtype=str))

        s3 = MagicMock()
        s3.get_paginator.return_value.paginate.return_value = [
            {"Contents": [{"Key": "feedback/pending/x.npz"}]}
        ]
        if read_error:
            s3.get_object.side_effect = RuntimeError("denied")
        else:
            s3.get_object.side_effect = lambda **_: {
                "Body": io.BytesIO(buffer.getvalue())
            }
        table = MagicMock()
        table.scan.return_value = {"Items": items}
        boto3 = types.SimpleNamespace(
            client=lambda *a, **k: s3,
            resource=lambda *a, **k: MagicMock(Table=lambda name: table),
        )
        monkeypatch.setitem(sys.modules, "boto3", boto3)
        until = NOW.isoformat()
        args = ["--output", str(tmp_path / "out.json"), "--until", until, "--write"]
        return script.main(args + ["--min-exposures", "10"]), s3

    def test_main_write_refused_when_guard_fails(self, monkeypatch, tmp_path):
        code, s3 = self._run_main(monkeypatch, tmp_path, read_error=True)
        assert code == 2
        s3.put_object.assert_not_called()
        assert (tmp_path / "out.json").exists()  # 로컬 결과는 남긴다

    def test_main_write_uploads_when_guard_passes(self, monkeypatch, tmp_path):
        code, s3 = self._run_main(monkeypatch, tmp_path)
        assert code == 0
        keys = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
        assert keys == ["engagement/archive/2026-11-02.json", "engagement/current.json"]
