# -*- coding: utf-8 -*-
"""추천 반응률 보너스 (engagement_v1) 테스트

- 집계: 계층 축소 공식, 노출 0, 중복 쌍, 위치 보정, 그룹 제외(얼굴형 재보정 이전)
- 서빙: 그룹 -> 스타일 -> mu 폴백, 보너스 상한, TTL 캐시, S3 실패 시 보너스 0
- 플래그 꺼짐 / 대조군 / 테이블 없음 응답이 기존(f3aa191) 스냅샷과 바이트 단위 동일
- 탐색 x 보너스 4가지 조합의 호출 k·순위·동점권·meta
- 사유 문구 조건, 엔드포인트 저장·로그 필드, 집계 스크립트 보조 함수

AWS 없이 돈다 (S3 로더·recommend_top_k 는 가짜).
"""

import io
import json
import os
from datetime import datetime, timedelta, timezone
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
    compute_bonus,
)
from services.rec_exploration import ARM_CONTROL as EXPLORE_CONTROL
from services.rec_exploration import ARM_EXPLORE, POLICY_DETERMINISTIC, POLICY_EXPLORE

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "rec_engagement_baseline.json"

MODEL_VERSION = "v6_feedback_20260909"
IMAGE_HASH = "e" * 64

# 스냅샷 생성 때와 같은 풀 (hairstyle_id = 100 + i)
SCORES = [
    62.4,
    62.1,
    61.9,
    61.7,
    61.2,
    60.8,
    60.5,
    60.1,
    59.6,
    59.0,
    58.2,
    57.5,
    56.9,
    56.0,
    55.1,
    54.3,
    53.0,
    52.2,
    51.0,
    50.0,
]
TRENDING = [
    {"style_name": "트렌드A", "reason": "요즘 인기", "source": "trending"},
    {"style_name": "트렌드B", "reason": "요즘 인기", "source": "trending"},
]
NOW = datetime(2026, 11, 2, tzinfo=timezone.utc)


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


def _payload(mu=0.5, styles=None, groups=None, version="eng_test_v1"):
    return {
        "schema_version": 1,
        "version": version,
        "mu": mu,
        "styles": styles or {},
        "groups": groups or {},
    }


def _entry(rate, exposures=100, reactions=0.0):
    return {"rate": rate, "exposures": exposures, "reactions": reactions}


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
    monkeypatch.setattr(settings, "REC_ENGAGEMENT_REASON_MIN_REACTIONS", 15.0)
    yield
    rec_engagement.reset_engagement_store(None)


def _install_table(payload):
    table = EngagementTable.from_payload(payload) if payload is not None else None
    store = _FakeStore(table)
    rec_engagement.reset_engagement_store(store)  # type: ignore[arg-type]
    return store


# 보너스로 스타일05(105)를 끌어올리고 스타일00(100)을 내리는 테이블
BOOST_PAYLOAD = _payload(
    mu=0.5,
    styles={"100": _entry(0.5), "105": _entry(0.6)},
    groups={
        "male|둥근형": {
            "100": _entry(0.2, 300),  # lift -0.6 -> -1.8
            "105": _entry(1.0, 120, 60.0),  # lift +1.0 -> +3.0 (상한)
        }
    },
)


# ========== 집계 ==========


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

    def test_hierarchical_rates_without_position_adjust(self):
        g = "male|둥근형"
        exposures = [eng.Exposure(f"a{i}", 1, 1, g) for i in range(10)] + [
            eng.Exposure(f"b{i}", 2, 1, None) for i in range(10)
        ]
        reactions = {("a0", 1): (3.0, "save"), ("a1", 1): (1.0, "synthesize")}
        result = eng.compute_engagement(
            exposures, reactions, a1=2.0, a2=1.0, position_adjust=False
        )
        mu = 4.0 / 20
        m1 = 2.0 / mu
        m2 = 1.0 / mu
        style1 = (4.0 + m1 * mu) / (10 + m1)
        style2 = (0.0 + m1 * mu) / (10 + m1)
        assert result["mu"] == pytest.approx(mu)
        assert result["m1"] == pytest.approx(m1)
        assert result["styles"]["1"]["rate"] == pytest.approx(style1, abs=1e-6)
        assert result["styles"]["2"]["rate"] == pytest.approx(style2, abs=1e-6)
        group = result["groups"][g]["1"]
        assert group["rate"] == pytest.approx((4.0 + m2 * style1) / (10 + m2), abs=1e-6)
        assert group["exposures"] == 10
        # group=None 노출은 스타일 통계에만
        assert "2" not in result["groups"][g]
        assert result["group_totals"][g]["exposures"] == 10

    def test_no_reactions_gives_mu_zero(self):
        exposures = [eng.Exposure("a", 1, 1, "male|계란형")]
        result = eng.compute_engagement(exposures, {})
        assert result["mu"] == 0.0
        assert result["m1"] == 0.0
        assert result["styles"]["1"]["rate"] == 0.0

    def test_position_factors_shrink_to_one(self):
        exposures = [eng.Exposure(f"a{i}", i, 1 + i % 3, None) for i in range(300)]
        # 1위만 반응
        reactions = {(f"a{i}", i): (1.0, "synthesize") for i in range(0, 300, 3)}
        factors = eng.position_factors(exposures, reactions, prior=20.0)
        assert factors[1] > 1.5
        assert factors[2] < 1.0 and factors[3] < 1.0
        # 사전 강도가 크면 1 에 가깝다
        strong = eng.position_factors(exposures, reactions, prior=1e6)
        assert all(abs(v - 1.0) < 0.01 for v in strong.values())

    def test_position_adjust_removes_rank_advantage(self):
        """늘 1위에 나오던 스타일이 위치 덕에 높은 반응률을 받지 않는다"""
        exposures, reactions = [], {}
        for i in range(600):
            rank = 1 + i % 3
            hid = {1: 1, 2: 2, 3: 3}[rank]
            exposures.append(eng.Exposure(f"a{i}", hid, rank, None))
            # 위치 효과만 있는 세계: 1위 30%, 2위 20%, 3위 10%
            if (i // 3) % 10 < {1: 3, 2: 2, 3: 1}[rank]:
                reactions[(f"a{i}", hid)] = (1.0, "synthesize")
        raw = eng.compute_engagement(
            exposures, reactions, a1=1.0, position_adjust=False
        )
        adjusted = eng.compute_engagement(
            exposures, reactions, a1=1.0, position_adjust=True, position_prior=1.0
        )
        raw_spread = raw["styles"]["1"]["rate"] / raw["styles"]["3"]["rate"]
        adj_spread = adjusted["styles"]["1"]["rate"] / adjusted["styles"]["3"]["rate"]
        assert raw_spread > 2.5
        assert adj_spread == pytest.approx(1.0, abs=0.1)


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
            {
                "analysis_id": "x",
                "hairstyle_id": 5,
                "source": "implicit",
                "signal_type": "share",
            }
        )
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
                {
                    "analysis_id": "x",
                    "hairstyle_id": 1,
                    "source": "implicit",
                    "signal_type": "?",
                }
            )
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
        exposures = [eng.Exposure("a", 1, 1, "male|계란형")]
        reactions = {("a", 1): (1.0, "synthesize"), ("old", 9): (3.0, "save")}
        result = eng.compute_engagement(exposures, reactions, position_adjust=False)
        assert result["summary"]["matched_pairs"] == 1
        assert result["summary"]["orphan_pairs"] == 1
        assert result["summary"]["orphan_by_kind"] == {"save": 1}
        assert result["summary"]["reaction_weight"] == 1.0


class TestExposures:
    def _item(self, created, cards, gender="male", face="둥근형", aid="a1"):
        return {
            "analysis_id": aid,
            "created_at": created,
            "gender": gender,
            "face_shape": face,
            "recommended_styles": cards,
        }

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
            (1, 1, "male|둥근형"),
            (2, 3, "male|둥근형"),
        ]
        assert stats.trending_exposures == 1
        assert stats.skipped_out_of_window == 1

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

    def test_payload_has_no_identifiers(self):
        exposures = [eng.Exposure("analysis-secret", 1, 1, "male|둥근형")]
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
        assert "analysis-secret" not in text
        assert "user_id" not in text and "device_id" not in text
        assert payload["version"] == "eng_20261102T000000Z"
        # 서빙 쪽이 그대로 읽을 수 있어야 한다
        table = EngagementTable.from_payload(json.loads(text))
        assert table.version == payload["version"]
        assert 1 in table.groups["male|둥근형"]


# ========== 서빙: 테이블 / 보너스 ==========


class TestTable:
    def test_group_falls_back_to_style_then_mu(self):
        table = EngagementTable.from_payload(BOOST_PAYLOAD)
        assert table.group_rate("male|둥근형", 105) == (1.0, 120)
        assert table.group_rate("female|둥근형", 105) == (0.6, 0)  # 그룹 없음 -> 스타일
        assert table.group_rate("male|둥근형", 999) == (0.5, 0)  # 스타일 없음 -> mu
        assert table.lift("female|계란형", 999) == 0.0

    def test_mu_zero_gives_zero_lift(self):
        table = EngagementTable.from_payload(
            _payload(mu=0.0, styles={"1": _entry(0.0)})
        )
        assert table.lift("male|둥근형", 1) == 0.0

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            [],
            {"schema_version": 99, "version": "v", "mu": 0.5},
            {"schema_version": 1, "mu": 0.5},
            {"schema_version": 1, "version": "v", "mu": "x"},
            _payload(styles={"1": {"rate": -1, "exposures": 1}}),
            _payload(styles={"abc": _entry(0.5)}),
        ],
    )
    def test_invalid_payload_raises(self, payload):
        with pytest.raises((ValueError, TypeError)):
            EngagementTable.from_payload(payload)

    def test_bonus_cap(self):
        assert compute_bonus(10.0, 3.0, 3.0) == 3.0
        assert compute_bonus(-10.0, 3.0, 3.0) == -3.0
        assert compute_bonus(0.5, 3.0, 3.0) == pytest.approx(1.5)
        assert compute_bonus(1.0, 3.0, -2.0) == 2.0  # 음수 상한도 절댓값으로

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

    def test_ttl_cache(self):
        clock = [0.0]
        body = json.dumps(BOOST_PAYLOAD).encode()
        load, calls = self._loader([body, json.dumps(_payload(version="v2")).encode()])
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
        load, calls = self._loader(
            [RuntimeError("s3 down"), json.dumps(BOOST_PAYLOAD).encode()]
        )
        store = EngagementStore(
            load, ttl_seconds=3600, retry_seconds=300, clock=lambda: clock[0]
        )
        assert store.get() is None
        clock[0] = 299
        assert store.get() is None
        assert len(calls) == 1  # 요청마다 S3 를 부르지 않는다
        clock[0] = 300
        assert store.get().version == "eng_test_v1"

    @pytest.mark.parametrize(
        "body", [b"{not json", b"[]", json.dumps({"schema_version": 1}).encode()]
    )
    def test_parse_error_gives_none(self, body):
        store = EngagementStore(lambda: body, ttl_seconds=60)
        assert store.get() is None

    def test_default_ttl_from_settings(self, monkeypatch):
        monkeypatch.setattr(settings, "REC_ENGAGEMENT_CACHE_TTL_SECONDS", 120)
        assert EngagementStore(lambda: b"").ttl_seconds == 120.0

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
        # 계측용 top_score 는 모델 점수
        assert meta["top_score"] == 62.1
        assert out == {
            "engagement_arm": ARM_TREATMENT,
            "engagement_version": "eng_test_v1",
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
            # 동점권은 보너스 포함 점수 기준: 63.8 - 3 = 60.8 이상 -> 105, 101, 102, 103, 104
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
        # 100 은 62.4 - 1.8 = 60.6 -> 스타일06(60.5) 바로 위로 내려간다
        names = [r["style_name"] for r in ranking]
        assert names.index("스타일00") > names.index("스타일05")
        assert {"style_name": "스타일00", "score": 0.61} in ranking
        assert len(ranking) == 20
        scores = [r["score"] for r in ranking[3:]]
        assert scores == sorted(scores, reverse=True)

    def test_pool_failure_falls_back_without_bonus(self, base_settings, monkeypatch):
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
        assert out == {}


# ========== 사유 문구 ==========


class TestReason:
    def _plan(
        self,
        payload,
        gender="male",
        face_shape="둥근형",
        seed=IMAGE_HASH,
        min_reactions=0.0,
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
            reason_min_reactions=min_reactions,
        )

    def test_reason_when_lift_and_exposure_enough(self):
        plan = self._plan(BOOST_PAYLOAD)
        reason = plan.reason(105)
        assert reason is not None
        assert "둥근형 남성" in reason
        assert reason in [
            t.format(who="둥근형 남성") for t in rec_engagement.REASON_TEMPLATES
        ]

    def test_unknown_gender_uses_face_shape_only(self):
        payload = _payload(groups={"unknown|둥근형": {"105": _entry(1.0, 80)}})
        reason = self._plan(payload, gender="neutral").reason(105)
        assert reason is not None
        assert "둥근형 얼굴" in reason
        assert "남성" not in reason and "여성" not in reason

    def test_min_reactions_required(self):
        """명시 좋아요 몇 개로 생긴 lift 로는 "많이 저장한" 문구가 나가지 않는다"""
        payload = _payload(
            groups={
                "male|둥근형": {"1": _entry(1.0, 100, 6.0), "2": _entry(1.0, 100, 15.0)}
            }
        )
        plan = self._plan(payload, min_reactions=15.0)
        assert plan.reason(1) is None
        assert plan.reason(2) is not None

    def test_no_reason_below_thresholds(self):
        payload = _payload(
            groups={
                "male|둥근형": {
                    "1": _entry(0.64, 500),  # lift 0.28 < 0.3
                    "2": _entry(1.0, 49),  # 노출 부족
                    "3": _entry(0.66, 50),  # lift 0.32, 노출 50 -> 문구
                }
            },
            styles={"4": _entry(1.0, 500)},  # 그룹 값 없음 -> 문구 없음
        )
        plan = self._plan(payload)
        assert plan.reason(1) is None
        assert plan.reason(2) is None
        assert plan.reason(3) is not None
        assert plan.reason(4) is None
        assert self._plan(_payload(mu=0.0)).reason(3) is None

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

    def test_reason_replaced_only_in_treatment(self, base_settings, monkeypatch):
        monkeypatch.setattr(settings, "REC_ENGAGEMENT_ENABLED", True)
        _install_table(BOOST_PAYLOAD)
        cards = _ml_cards(_recommend(_service()))
        assert "둥근형 남성" in cards[0]["reason"]  # 105
        assert all(c["reason"].startswith("ML 모델 추천") for c in cards[1:])

        monkeypatch.setattr(settings, "REC_ENGAGEMENT_RATIO", 0.0)
        control = _ml_cards(_recommend(_service()))
        assert all("사용자" not in c["reason"] for c in control)


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
        assert "experiment_id" not in saved  # 탐색 실험 필드와 섞이지 않는다
        event = _analysis_complete(log)
        assert event["engagement_arm"] == ARM_CONTROL
        assert event["engagement_version"] == "eng_test_v1"

        off_saved = off_save.call_args.kwargs["analysis_result"]
        assert "engagement_arm" not in off_saved
        off_event = _analysis_complete(off_log)
        assert off_event["engagement_arm"] is None
        assert off_event["engagement_version"] is None

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
        assert saved["experiment_id"] == "rec_explore_v1"
        assert saved["ab_variant"] == ARM_EXPLORE
        saved_cards = _ml_cards({"recommendations": saved["recommendations"]})
        assert all("engagement_bonus" in c for c in saved_cards)
        event = _analysis_complete(log)
        assert event["engagement_arm"] == ARM_TREATMENT
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
                },
                processing_time=0.1,
                detection_method="ml",
            )
        data = save.call_args.args[0]
        assert data["engagement_arm"] == ARM_TREATMENT
        assert data["engagement_version"] == "eng_x"

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
                }
            )
            conn.save_analysis({"face_shape": "둥근형"})
        first = table.put_item.call_args_list[0].kwargs["Item"]
        second = table.put_item.call_args_list[1].kwargs["Item"]
        assert first["engagement_arm"] == ARM_CONTROL
        assert first["engagement_version"] == "eng_x"
        assert "engagement_arm" not in second and "engagement_version" not in second


# ========== 집계 스크립트 ==========


class TestScript:
    def test_parse_weights(self):
        from scripts.build_style_engagement import parse_weights

        weights = parse_weights("save=5, explicit_good=0")
        assert weights["save"] == 5.0
        assert weights["explicit_good"] == 0.0
        assert weights["share"] == 4.0
        import argparse

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
                {
                    "analysis_id": "a",
                    "hairstyle_id": 1,
                    "source": "implicit",
                    "signal_type": "save",
                }
            ),
            "feedback/processed/batch_x/a_2.npz": npz(
                {
                    "analysis_id": "a",
                    "hairstyle_id": 2,
                    "source": "implicit",
                    "signal_type": "share",
                }
            ),
            "feedback/processed/batch_x/old.npz": npz(
                {"analysis_id": "b", "hairstyle_id": 3, "feedback": "good"}
            ),
            "feedback/pending/broken.npz": b"xx",
        }
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
            ("a", 2, "share"),
            ("b", 3, "explicit_good"),
        ]
        assert stats["no_metadata"] == 1

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
        assert "user_id" not in first["ExpressionAttributeNames"].values()
        assert first["ExpressionAttributeValues"] == {":since": "2026-11-02"}
        assert table.scan.call_args_list[1].kwargs["ExclusiveStartKey"] == {
            "analysis_id": "a"
        }
        assert not table.put_item.called
