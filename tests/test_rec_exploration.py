# -*- coding: utf-8 -*-
"""추천 Top-3 동점권 탐색 (explore_v1) 테스트

- 플래그 꺼짐: 기존 k=3 호출 경로 그대로, 노출 메타는 결정적(propensity 1.0)
- 동점권 경계·채우기, 시드 재현성, 표시 순서, 포함 확률
- 실험군/대조군 배정과 meta / analysis_complete 로그 필드
- 트렌드 카드는 탐색과 무관, full_ranking 은 실험군에서만 화면 Top-3 를 앞에 둔다

AWS·실제 모델 없이 돈다 (recommend_top_k 는 점수 리스트를 돌려주는 mock).
"""

import io
import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from config.settings import settings
from services.rec_exploration import (
    ARM_CONTROL,
    ARM_EXPLORE,
    POLICY_DETERMINISTIC,
    POLICY_EXPLORE,
    assign_arm,
    explore_select,
    inclusion_probabilities,
    selection_seed,
    tie_band_size,
)

MODEL_VERSION = "v6_feedback_20260909"

# 9월 실측처럼 1위 근처가 촘촘한 풀 (원점수, 내림차순)
TIGHT_SCORES = [
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


def _ranked(scores):
    return [
        {
            "hairstyle_id": i,
            "hairstyle": f"스타일{i:02d}",
            "score": round(s, 2),
            "original_score": s,
        }
        for i, s in enumerate(scores)
    ]


def _service(ranked, model_version=MODEL_VERSION):
    from services.hybrid_recommender import MLRecommendationService

    with patch.object(MLRecommendationService, "__init__", return_value=None):
        service = MLRecommendationService()
    service.ml_available = True
    service.reason_generator = None
    service.ml_recommender = MagicMock()
    service.ml_recommender.allowed_indices = None
    service.ml_recommender.model_version = model_version
    service.ml_recommender.recommend_top_k.side_effect = lambda **kw: [
        dict(r) for r in ranked[: kw["k"]]
    ]
    return service


def _recommend(service, image_hash="a" * 64, user_id=None, trending=None):
    with patch(
        "services.trending_style_service.get_trending_style_service"
    ) as get_trending:
        get_trending.return_value.pick_trending.return_value = [
            dict(t) for t in (trending or [])
        ]
        return service.recommend(
            image_data=b"",
            face_shape="계란형",
            skin_tone="봄웜",
            face_features=[1.4, 120.0, 130.0, 110.0, 0.92, 0.85],
            skin_features=[45.0, 15.0],
            gender="male",
            image_hash=image_hash,
            user_id=user_id,
        )


def _ml_cards(result):
    return [r for r in result["recommendations"] if r.get("source") == "ml"]


@pytest.fixture
def explore_on(monkeypatch):
    monkeypatch.setattr(settings, "REC_EXPLORE_ENABLED", True)
    monkeypatch.setattr(settings, "REC_EXPLORE_RATIO", 1.0)  # 전원 실험군
    monkeypatch.setattr(settings, "REC_EXPLORE_POOL", 20)
    monkeypatch.setattr(settings, "REC_EXPLORE_MARGIN", 3.0)
    monkeypatch.setattr(settings, "REC_EXPLORE_TEMPERATURE", 1.5)


# ========== 플래그 꺼짐 ==========


class TestFlagOff:
    def test_default_is_off(self):
        from config.settings import Settings

        assert Settings.model_fields["REC_EXPLORE_ENABLED"].default is False

    def test_top3_call_is_unchanged(self, monkeypatch):
        monkeypatch.setattr(settings, "REC_EXPLORE_ENABLED", False)
        monkeypatch.setattr(settings, "FULL_RANKING_LIMIT", 0)
        service = _service(_ranked(TIGHT_SCORES))

        result = _recommend(service)

        mock = service.ml_recommender.recommend_top_k
        assert mock.call_count == 1
        assert mock.call_args.kwargs == {
            "face_shape": "계란형",
            "skin_tone": "봄웜",
            "k": 3,
            "face_features": [1.4, 120.0, 130.0, 110.0, 0.92, 0.85],
            "skin_features": [45.0, 15.0],
            "gender": "male",
        }
        cards = _ml_cards(result)
        assert [c["style_name"] for c in cards] == ["스타일00", "스타일01", "스타일02"]
        assert [c["score"] for c in cards] == [0.62, 0.62, 0.62]

    def test_exposure_meta_is_deterministic(self, monkeypatch):
        monkeypatch.setattr(settings, "REC_EXPLORE_ENABLED", False)
        result = _recommend(_service(_ranked(TIGHT_SCORES)))

        cards = _ml_cards(result)
        assert [c["rank"] for c in cards] == [1, 2, 3]
        assert [c["ml_rank"] for c in cards] == [1, 2, 3]
        assert [c["propensity"] for c in cards] == [1.0, 1.0, 1.0]
        assert [c["raw_score"] for c in cards] == [62.4, 62.1, 61.9]
        meta = result["meta"]
        assert meta["rec_policy"] == POLICY_DETERMINISTIC
        assert meta["experiment_arm"] is None
        assert meta["tie_band_size"] is None
        assert meta["margin"] is None
        assert meta["temperature"] is None

    def test_flag_on_without_image_hash_stays_deterministic(self, explore_on):
        service = _service(_ranked(TIGHT_SCORES))
        result = _recommend(service, image_hash=None)

        assert service.ml_recommender.recommend_top_k.call_args_list[0].kwargs["k"] == 3
        assert result["meta"]["rec_policy"] == POLICY_DETERMINISTIC
        assert result["meta"]["experiment_arm"] is None


# ========== 동점권 ==========


class TestTieBand:
    def test_boundary_is_inclusive(self):
        # top - margin = 87.0 는 포함, 86.99 는 제외
        assert tie_band_size([90.0, 89.0, 88.0, 87.0, 86.99], 3.0) == 4

    def test_float_boundary(self):
        # 0.1 단위 부동소수 오차가 있어도 경계 포함
        assert tie_band_size([62.4, 61.0, 59.4, 59.39], 3.0) == 3

    def test_fills_by_score_when_band_under_3(self):
        assert tie_band_size([90.0, 80.0, 70.0, 60.0], 3.0) == 3
        selection = explore_select([90.0, 80.0, 70.0, 60.0], 3, 3.0, 1.5, seed=1)
        assert selection.indices == [0, 1, 2]
        assert selection.propensities == [1.0, 1.0, 1.0]
        assert selection.band_size == 3

    def test_small_pool(self):
        assert tie_band_size([90.0, 80.0], 3.0) == 2
        assert tie_band_size([], 3.0) == 0
        selection = explore_select([90.0, 80.0], 3, 3.0, 1.5, seed=1)
        assert selection.indices == [0, 1]
        assert selection.propensities == [1.0, 1.0]

    def test_tight_scores_band(self):
        # 62.4 - 3.0 = 59.4 이상: 62.4 ~ 59.6 의 9개
        assert tie_band_size(TIGHT_SCORES, 3.0) == 9


# ========== 확률 선택 ==========


class TestExploreSelect:
    def test_same_seed_same_result(self):
        seed = selection_seed("b" * 64, MODEL_VERSION)
        first = explore_select(TIGHT_SCORES, 3, 3.0, 1.5, seed)
        for _ in range(5):
            assert explore_select(TIGHT_SCORES, 3, 3.0, 1.5, seed) == first

    def test_seed_depends_on_hash_and_model_version(self):
        assert selection_seed("b" * 64, "v1") != selection_seed("c" * 64, "v1")
        assert selection_seed("b" * 64, "v1") != selection_seed("b" * 64, "v2")

    def test_different_hashes_spread_over_band(self):
        band = tie_band_size(TIGHT_SCORES, 3.0)
        top_picks = {}
        chosen_sets = set()
        for i in range(400):
            seed = selection_seed(f"{i:064x}", MODEL_VERSION)
            selection = explore_select(TIGHT_SCORES, 3, 3.0, 1.5, seed)
            assert all(idx < band for idx in selection.indices)
            chosen_sets.add(tuple(selection.indices))
            top = selection.indices[0]
            top_picks[top] = top_picks.get(top, 0) + 1

        assert len(chosen_sets) > 20
        # 결정적이면 1위가 항상 0번. 표시는 점수순이라 0번이 뽑히면 늘 1위지만
        # (이 풀에서 포함 확률 약 0.55) 나머지는 다른 스타일이 1위가 된다
        assert top_picks[0] / 400 < 0.7
        assert len(top_picks) >= 4

    def test_display_order_is_score_descending(self):
        for i in range(50):
            selection = explore_select(
                TIGHT_SCORES, 3, 3.0, 1.5, selection_seed(f"{i:064x}", "v")
            )
            assert selection.indices == sorted(selection.indices)
            shown = [TIGHT_SCORES[idx] for idx in selection.indices]
            assert shown == sorted(shown, reverse=True)

    def test_does_not_touch_global_random_state(self):
        import random

        import numpy as np

        random.seed(123)
        np.random.seed(123)
        expected = (random.random(), np.random.random())
        random.seed(123)
        np.random.seed(123)
        explore_select(TIGHT_SCORES, 3, 3.0, 1.5, seed=7)
        assert (random.random(), np.random.random()) == expected


# ========== 포함 확률 ==========


class TestPropensity:
    def test_band_inclusion_probabilities_sum_to_3(self):
        band = TIGHT_SCORES[: tie_band_size(TIGHT_SCORES, 3.0)]
        probs = inclusion_probabilities(band, 3, 1.5, seed=42)
        assert all(0.0 <= p <= 1.0 for p in probs)
        assert sum(probs) == pytest.approx(3.0, abs=1e-9)
        # 점수가 높을수록 포함 확률이 높다 (몬테카를로 오차 허용)
        assert probs[0] > probs[-1]

    def test_uniform_scores_have_equal_probabilities(self):
        probs = inclusion_probabilities([70.0] * 6, 3, 1.5, seed=3)
        for p in probs:
            assert p == pytest.approx(0.5, abs=0.05)

    def test_selected_propensities_in_range(self):
        for i in range(30):
            selection = explore_select(
                TIGHT_SCORES, 3, 3.0, 1.5, selection_seed(f"{i:064x}", "v")
            )
            assert all(0.0 < p <= 1.0 for p in selection.propensities)
            assert all(p < 1.0 for p in selection.propensities)


# ========== 실험 배정 ==========


class TestAssignment:
    def test_split_is_roughly_half(self):
        arms = [assign_arm(None, f"{i:064x}", 0.5) for i in range(2000)]
        share = arms.count(ARM_EXPLORE) / len(arms)
        assert 0.45 < share < 0.55

    def test_user_id_takes_precedence_over_image(self):
        arms = {assign_arm("user-1", f"{i:064x}", 0.5) for i in range(50)}
        assert len(arms) == 1

    def test_ratio_bounds(self):
        assert assign_arm(None, "a" * 64, 0.0) == ARM_CONTROL
        assert assign_arm(None, "a" * 64, 1.0) == ARM_EXPLORE


# ========== 서비스 통합 (mock 모델) ==========


class TestServiceExplore:
    def test_explore_arm_uses_pool_and_records_exposure(self, explore_on):
        service = _service(_ranked(TIGHT_SCORES))
        result = _recommend(service)

        first_call = service.ml_recommender.recommend_top_k.call_args_list[0]
        assert first_call.kwargs["k"] == 20
        assert first_call.kwargs["log_selection"] is False

        cards = _ml_cards(result)
        assert len(cards) == 3
        assert [c["rank"] for c in cards] == [1, 2, 3]
        raw = [c["raw_score"] for c in cards]
        assert raw == sorted(raw, reverse=True)
        assert all(c["raw_score"] >= 62.4 - 3.0 for c in cards)
        ml_ranks = [c["ml_rank"] for c in cards]
        assert ml_ranks == sorted(ml_ranks)
        assert all(1 <= r <= 9 for r in ml_ranks)
        # ml_rank 은 풀의 결정적 순위 - 같은 이름의 스타일이 그 자리에 있어야 한다
        for c in cards:
            assert c["style_name"] == f"스타일{c['ml_rank'] - 1:02d}"
        assert all(0.0 < c["propensity"] < 1.0 for c in cards)

        meta = result["meta"]
        assert meta["rec_policy"] == POLICY_EXPLORE
        assert meta["experiment_arm"] == ARM_EXPLORE
        assert meta["tie_band_size"] == 9
        assert meta["margin"] == 3.0
        assert meta["temperature"] == 1.5
        assert meta["ml_count"] == 3
        # top_score 는 표시된 카드의 원점수 기준
        assert meta["top_score"] == raw[0]

    def test_same_image_same_cards(self, explore_on):
        service = _service(_ranked(TIGHT_SCORES))
        first = _ml_cards(_recommend(service, image_hash="d" * 64))
        second = _ml_cards(_recommend(service, image_hash="d" * 64))
        assert first == second

    def test_control_arm_is_deterministic_top3_with_band(self, explore_on, monkeypatch):
        monkeypatch.setattr(settings, "REC_EXPLORE_RATIO", 0.0)  # 전원 대조군
        service = _service(_ranked(TIGHT_SCORES))
        result = _recommend(service)

        cards = _ml_cards(result)
        assert [c["style_name"] for c in cards] == ["스타일00", "스타일01", "스타일02"]
        assert [c["ml_rank"] for c in cards] == [1, 2, 3]
        assert [c["propensity"] for c in cards] == [1.0, 1.0, 1.0]
        meta = result["meta"]
        assert meta["rec_policy"] == POLICY_DETERMINISTIC
        assert meta["experiment_arm"] == ARM_CONTROL
        assert meta["tie_band_size"] == 9

    def test_control_matches_flag_off_cards(self, explore_on, monkeypatch):
        ranked = _ranked(TIGHT_SCORES)
        monkeypatch.setattr(settings, "REC_EXPLORE_RATIO", 0.0)
        control = _ml_cards(_recommend(_service(ranked)))
        monkeypatch.setattr(settings, "REC_EXPLORE_ENABLED", False)
        off = _ml_cards(_recommend(_service(ranked)))
        assert control == off

    def test_duplicate_names_in_pool_keep_3_cards(self, explore_on):
        ranked = _ranked(TIGHT_SCORES)
        ranked[1]["hairstyle"] = "스타일 00"  # 띄어쓰기 중복
        cards = _ml_cards(_recommend(_service(ranked)))
        assert len(cards) == 3
        assert len({c["style_name"].replace(" ", "") for c in cards}) == 3

    def test_pool_failure_falls_back_to_deterministic(self, explore_on, monkeypatch):
        monkeypatch.setattr(settings, "FULL_RANKING_LIMIT", 0)
        ranked = _ranked(TIGHT_SCORES)
        service = _service(ranked)
        service.ml_recommender.recommend_top_k.side_effect = [
            RuntimeError("pool boom"),
            [dict(r) for r in ranked[:3]],
        ]
        result = _recommend(service)

        assert [c["style_name"] for c in _ml_cards(result)] == [
            "스타일00",
            "스타일01",
            "스타일02",
        ]
        assert service.ml_recommender.recommend_top_k.call_args_list[1].kwargs["k"] == 3
        assert result["meta"]["rec_policy"] == POLICY_DETERMINISTIC
        assert result["meta"]["experiment_arm"] == ARM_EXPLORE

    def test_trending_cards_follow_ml_without_exposure_meta(self, explore_on):
        trending = [
            {"style_name": "트렌드A", "reason": "요즘 인기", "source": "trending"},
            {"style_name": "트렌드B", "reason": "요즘 인기", "source": "trending"},
        ]
        result = _recommend(_service(_ranked(TIGHT_SCORES)), trending=trending)

        recs = result["recommendations"]
        assert [r["source"] for r in recs] == ["ml", "ml", "ml", "trending", "trending"]
        # 트렌드 카드 rank 는 ML 카드 뒤로 연속 (4, 5)
        assert [r["rank"] for r in recs] == [1, 2, 3, 4, 5]
        for r in recs[3:]:
            assert r.get("propensity") is None
            assert r.get("ml_rank") is None
            assert r.get("raw_score") is None
        assert result["meta"]["trending_count"] == 2

    def test_trending_rank_flag_off(self, monkeypatch):
        monkeypatch.setattr(settings, "REC_EXPLORE_ENABLED", False)
        trending = [
            {"style_name": "트렌드A", "reason": "요즘 인기", "source": "trending"},
            {"style_name": "트렌드B", "reason": "요즘 인기", "source": "trending"},
        ]
        recs = _recommend(_service(_ranked(TIGHT_SCORES)), trending=trending)[
            "recommendations"
        ]
        assert [r["rank"] for r in recs] == [1, 2, 3, 4, 5]

    def test_full_ranking_deterministic_paths_unchanged(self, monkeypatch):
        """꺼짐·대조군은 full_ranking 이 점수순 그대로 (탐색 도입 전과 동일)"""
        ranked = _ranked(TIGHT_SCORES)
        expected = [
            {"style_name": r["hairstyle"], "score": round(r["score"] / 100.0, 2)}
            for r in ranked
        ]

        monkeypatch.setattr(settings, "REC_EXPLORE_ENABLED", False)
        off_service = _service(ranked)
        off = _recommend(off_service)["full_ranking"]

        monkeypatch.setattr(settings, "REC_EXPLORE_ENABLED", True)
        monkeypatch.setattr(settings, "REC_EXPLORE_RATIO", 0.0)  # 전원 대조군
        control_service = _service(ranked)
        control_result = _recommend(control_service)
        control = control_result["full_ranking"]

        assert control_result["meta"]["experiment_arm"] == ARM_CONTROL
        assert off == expected
        assert control == expected
        off_call = off_service.ml_recommender.recommend_top_k.call_args_list[-1]
        control_call = control_service.ml_recommender.recommend_top_k.call_args_list[-1]
        assert control_call.kwargs == off_call.kwargs

    def test_full_ranking_follows_displayed_top3_in_explore_arm(self, explore_on):
        ranked = _ranked(TIGHT_SCORES)
        for i in range(30):
            result = _recommend(_service(ranked), image_hash=f"{i:064x}")
            cards = _ml_cards(result)
            ranking = result["full_ranking"]
            shown = [c["style_name"] for c in cards]

            assert [r["style_name"] for r in ranking[:3]] == shown
            rest = [r["style_name"] for r in ranking[3:]]
            assert rest == [
                r["hairstyle"] for r in ranked if r["hairstyle"] not in shown
            ]
            assert len(ranking) == len(ranked)
            # 점수는 전체 랭킹 값 그대로 유지
            by_name = {r["hairstyle"]: round(r["score"] / 100.0, 2) for r in ranked}
            assert all(r["score"] == by_name[r["style_name"]] for r in ranking)

    def test_full_ranking_empty_stays_empty_in_explore_arm(
        self, explore_on, monkeypatch
    ):
        monkeypatch.setattr(settings, "FULL_RANKING_LIMIT", 0)
        assert _recommend(_service(_ranked(TIGHT_SCORES)))["full_ranking"] == []

    def test_align_full_ranking_adds_missing_displayed_style(self):
        from services.hybrid_recommender import _align_full_ranking

        ranking = [
            {"style_name": "A", "score": 0.6},
            {"style_name": "B 컷", "score": 0.5},
            {"style_name": "C", "score": 0.4},
        ]
        recs = [
            {"style_name": "B컷", "source": "ml", "score": 0.5},
            {"style_name": "Z", "source": "ml", "score": 0.3},
            {"style_name": "C", "source": "trending"},
        ]
        assert _align_full_ranking(ranking, recs) == [
            {"style_name": "B 컷", "score": 0.5},
            {"style_name": "Z", "score": 0.3},
            {"style_name": "A", "score": 0.6},
            {"style_name": "C", "score": 0.4},
        ]

    def test_cards_convert_to_decimal_for_dynamodb(self, explore_on):
        from database.dynamodb_connection import _convert_floats_to_decimal

        cards = _ml_cards(_recommend(_service(_ranked(TIGHT_SCORES))))
        converted = _convert_floats_to_decimal(cards)
        for card in converted:
            assert isinstance(card["raw_score"], Decimal)
            assert isinstance(card["propensity"], Decimal)
            assert isinstance(card["ml_rank"], int)


# ========== 엔드포인트 (analysis_complete / 저장) ==========


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color="white").save(buffer, format="JPEG")
    return buffer.getvalue()


def _post(service):
    from core.dependencies import get_face_detection_service, get_hybrid_service
    from main import app

    features = MagicMock(
        face_shape="계란형",
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
            "api.endpoints.analyze.save_to_database", return_value=None
        ) as save, patch(
            "api.endpoints.analyze.log_structured"
        ) as log, patch(
            "api.endpoints.analyze.limiter.enabled", False
        ), TestClient(
            app
        ) as client:
            trending.return_value.pick_trending.return_value = []
            response = client.post(
                "/api/v2/analyze-hybrid",
                files={"file": ("face.jpg", _jpeg_bytes(), "image/jpeg")},
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


class TestEndpoint:
    def test_explore_fields_in_log_response_and_saved_cards(self, explore_on):
        response, save, log = _post(_service(_ranked(TIGHT_SCORES)))

        assert response.status_code == 200, response.text
        data = response.json()["data"]
        cards = _ml_cards(data)
        assert all(
            "propensity" in c and "ml_rank" in c and "raw_score" in c for c in cards
        )

        saved = save.call_args.kwargs["analysis_result"]["recommendations"]
        assert [c["propensity"] for c in _ml_cards({"recommendations": saved})] == [
            c["propensity"] for c in cards
        ]

        saved_result = save.call_args.kwargs["analysis_result"]
        assert saved_result["experiment_id"] == "rec_explore_v1"
        assert saved_result["ab_variant"] == ARM_EXPLORE

        event = _analysis_complete(log)
        assert event["rec_policy"] == POLICY_EXPLORE
        assert event["experiment_arm"] == ARM_EXPLORE
        assert event["tie_band_size"] == 9
        assert event["top_ml_rank"] == cards[0]["ml_rank"]
        assert event["top_style"] == cards[0]["style_name"]

    def test_flag_off_log_fields(self, monkeypatch):
        monkeypatch.setattr(settings, "REC_EXPLORE_ENABLED", False)
        response, save, log = _post(_service(_ranked(TIGHT_SCORES)))

        assert response.status_code == 200, response.text
        event = _analysis_complete(log)
        assert event["rec_policy"] == POLICY_DETERMINISTIC
        assert event["experiment_arm"] is None
        assert event["tie_band_size"] is None
        assert event["top_ml_rank"] == 1
        saved_result = save.call_args.kwargs["analysis_result"]
        assert "experiment_id" not in saved_result
        assert "ab_variant" not in saved_result

    def test_repository_passes_arm_to_dynamodb(self):
        from database.dynamodb_repository import DynamoDBAnalysisRepository

        with patch(
            "database.dynamodb_repository.save_analysis", return_value="id-1"
        ) as save:
            DynamoDBAnalysisRepository().save_analysis(
                image_hash="h",
                analysis_result={
                    "analysis": {"face_shape": "계란형", "personal_color": "봄웜"},
                    "recommendations": [],
                    "experiment_id": "rec_explore_v1",
                    "ab_variant": ARM_CONTROL,
                },
                processing_time=0.1,
                detection_method="ml",
            )
        data = save.call_args.args[0]
        assert data["experiment_id"] == "rec_explore_v1"
        assert data["ab_variant"] == ARM_CONTROL
