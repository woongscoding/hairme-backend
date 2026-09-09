"""
추천 후보 노이즈 스타일 제외 필터 테스트

style_embeddings.npz의 styles 배열에는 LLM 학습 데이터의 "피해야 할 스타일" 설명이
스타일명으로 잘못 흡수된 항목이 섞여 있다. 이 항목들이 추천 결과(top-k)에 노출되지
않는지, 그리고 제외 필터가 hairstyle_id(npz 원본 인덱스)를 깨뜨리지 않는지 검증한다.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from models.ml_recommender import (
    EXCLUDED_STYLE_PATTERNS,
    EXCLUDED_STYLES_PATH,
    MLHairstyleRecommender,
    _load_excluded_styles,
    is_excluded_style,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = PROJECT_ROOT / "data_source" / "style_embeddings.npz"

# 사용자 리포트에 등장한 실제 노이즈 스타일명 (npz 원본 표기 그대로)
KNOWN_NOISY_NAMES = [
    "포마드 (과도한 볼륨)",
    "쉼표머리 (과도한 컬)",
    "가르마펌 (과도한 컬)",
    "리젠트 컷 (과도한 볼륨)",
    "블레이드 펌 (과도한 컬)",
    "장발 히피펌: 과도한 볼륨과 컬은 얼굴을 더 넓어 보이게 만들 수 있습니다.",
    "울프컷 (나이들어 보일 수 있음)",
    "스왓컷: 너무 짧은 머리는 계란형 얼굴의 균형을 깨뜨릴 수 있습니다.",
]

# 정상 스타일명 (절대 제외되면 안 됨)
KNOWN_GOOD_NAMES = [
    "포마드",
    "크롭컷",
    "리프컷",
    "울프컷",
    "히피펌",
    "스왓컷",
    "댄디 컷",
    "볼륨펌",
    "시스루 뱅",
    "소프트 투블럭",
    # 비율 표기(5:5, 6:4)는 콜론이 있어도 문장 조각이 아니므로 유지되어야 한다
    "5:5 가르마 펌",
    "가르마 스타일 (6:4 또는 7:3)",
    "포마드 스타일 (7:3 가르마)",
    "장발 스타일 (5:5 가르마)",
]

FACE_FEATURES = [1.20, 458.0, 561.0, 447.0, 0.82, 0.80]
SKIN_FEATURES = [79.9, 12.1]


@pytest.fixture(scope="module")
def catalog():
    """npz 원본 스타일 카탈로그 (styles, embeddings)"""
    data = np.load(str(EMBEDDINGS_PATH), allow_pickle=False)
    return data["styles"].tolist(), data["embeddings"]


@pytest.fixture(scope="module")
def denylist():
    return _load_excluded_styles()


class _FakeAffinityModel:
    """
    타깃 임베딩과의 코사인 유사도를 점수로 반환하는 더미 모델

    타깃 스타일이 항상 1위가 되므로, 필터가 없으면 노이즈 스타일이
    top-1에 오른다는 것을 검증할 수 있다.
    """

    def __init__(self, target_embedding: np.ndarray):
        self.target = target_embedding / (np.linalg.norm(target_embedding) + 1e-9)

    def __call__(self, face_tensor, skin_tensor, style_tensor):
        styles = style_tensor.cpu().numpy()
        norms = np.linalg.norm(styles, axis=1, keepdims=True) + 1e-9
        scores = (styles / norms) @ self.target
        return torch.FloatTensor(scores)


def _make_recommender(styles, embeddings, target_name, allowed_indices=None):
    """__init__(모델 로딩)을 건너뛴 테스트용 추천기 인스턴스 생성"""
    inst = object.__new__(MLHairstyleRecommender)
    inst.device = torch.device("cpu")
    inst.styles = list(styles)
    inst.embeddings = embeddings
    inst.style_to_idx = {s: i for i, s in enumerate(styles)}
    inst.gender_metadata = {}
    inst.is_normalized_model = False
    inst.model_version = "v6"
    inst.attention_type = "multi_token"
    inst.sentence_model = None
    inst.excluded_styles = _load_excluded_styles()
    if allowed_indices is None:
        inst.allowed_indices = [
            i
            for i, name in enumerate(styles)
            if not is_excluded_style(name, inst.excluded_styles)
        ]
    else:
        inst.allowed_indices = allowed_indices
    inst.model = _FakeAffinityModel(embeddings[styles.index(target_name)])
    return inst


# ========== 1. 규칙 기반 필터 ==========


class TestExclusionRules:
    def test_known_noisy_names_are_excluded(self, denylist):
        for name in KNOWN_NOISY_NAMES:
            assert is_excluded_style(name, denylist), f"제외되지 않음: {name}"

    def test_known_good_names_are_kept(self, denylist):
        for name in KNOWN_GOOD_NAMES:
            assert not is_excluded_style(name, denylist), f"잘못 제외됨: {name}"

    def test_ratio_colon_is_not_treated_as_sentence(self, denylist):
        """비율 표기(5:5 / 6:4)는 문장 조각 패턴에 걸리지 않아야 한다"""
        assert not is_excluded_style("가르마 펌 (5:5, 6:4)", denylist)
        assert is_excluded_style("가일컷: 이마를 살짝 드러내어", denylist)

    def test_empty_name_is_excluded(self, denylist):
        assert is_excluded_style("", denylist)
        assert is_excluded_style("   ", denylist)

    def test_patterns_cover_required_rules(self):
        """요구 규칙(과도한 / 콜론+공백 / 습니다 / 수 있 / 마침표 종결) 모두 존재"""
        probes = {
            "테스트 (과도한 볼륨)": "과도한",
            "테스트: 설명 조각": "콜론+공백",
            "테스트 어울립니다": "습니다",
            "테스트 스타일입니다": "종결어미(입니다)",
            "테스트 연출합니다": "종결어미(합니다)",
            "테스트 (보일 수 있음)": "수 있",
            "테스트 스타일이다.": "마침표 종결",
        }
        for probe, label in probes.items():
            assert any(
                p.search(probe) for p in EXCLUDED_STYLE_PATTERNS
            ), f"{label} 규칙 미적용: {probe}"

    def test_denylist_alone_excludes_without_regex_match(self, denylist):
        """정규식에 안 걸리는 이름도 denylist에 있으면 제외된다"""
        name = "포마드 스타일 (너무 과한 볼륨)"
        assert not any(p.search(name) for p in EXCLUDED_STYLE_PATTERNS)
        assert is_excluded_style(name, denylist)
        assert not is_excluded_style(name, set())


# ========== 2. denylist 데이터 파일 ==========


class TestDenylistFile:
    def test_file_exists_and_has_comment(self):
        path = Path(EXCLUDED_STYLES_PATH)
        assert path.exists(), f"denylist 파일 없음: {path}"
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert "_comment" in doc and doc["_comment"].strip()
        assert isinstance(doc["excluded_styles"], list)

    def test_entries_exist_in_catalog_and_are_unique(self, catalog, denylist):
        styles, _ = catalog
        assert denylist, "denylist가 비어 있음"
        unknown = sorted(denylist - set(styles))
        assert not unknown, f"npz에 없는 이름: {unknown}"
        raw = json.loads(Path(EXCLUDED_STYLES_PATH).read_text(encoding="utf-8"))
        names = raw["excluded_styles"]
        assert len(names) == len(set(names)), "denylist에 중복 항목 존재"

    def test_missing_file_returns_empty_set(self, tmp_path):
        assert _load_excluded_styles(str(tmp_path / "nope.json")) == set()

    def test_broken_file_returns_empty_set(self, tmp_path):
        bad = tmp_path / "broken.json"
        bad.write_text("{not json", encoding="utf-8")
        assert _load_excluded_styles(str(bad)) == set()


# ========== 3. 후보 인덱스 구성 ==========


class TestCandidateIndices:
    def test_indices_are_original_catalog_positions(self, catalog):
        styles, embeddings = catalog
        rec = _make_recommender(styles, embeddings, "포마드")
        indices = rec._build_candidate_indices(k=3)

        assert indices == sorted(indices), "인덱스가 오름차순이 아님"
        assert len(set(indices)) == len(indices)
        for idx in indices:
            # 필터링된 위치가 아니라 npz 원본 인덱스여야 한다
            assert 0 <= idx < len(styles)
            assert not is_excluded_style(styles[idx], rec.excluded_styles)

    def test_noisy_styles_are_dropped_from_candidates(self, catalog):
        styles, embeddings = catalog
        rec = _make_recommender(styles, embeddings, "포마드")
        indices = set(rec._build_candidate_indices(k=3))

        assert len(indices) < len(styles), "제외된 후보가 하나도 없음"
        for name in KNOWN_NOISY_NAMES:
            assert styles.index(name) not in indices, f"후보에 남아있음: {name}"

    def test_fallback_when_candidates_below_k(self, caplog):
        """전부 노이즈라 k개를 못 채우면 필터를 포기하고 전체 후보로 폴백"""
        noisy = [
            "포마드 (과도한 볼륨)",
            "쉼표머리 (과도한 컬)",
            "리젠트 컷 (과도한 볼륨)",
        ]
        rec = object.__new__(MLHairstyleRecommender)
        rec.styles = noisy
        rec.excluded_styles = set()
        rec.allowed_indices = []

        with caplog.at_level(logging.WARNING, logger="models.ml_recommender"):
            indices = rec._build_candidate_indices(k=3)

        assert indices == [0, 1, 2], "전체 후보로 폴백되지 않음"
        assert any("폴백" in r.message for r in caplog.records)

    def test_no_fallback_when_enough_candidates(self, catalog):
        styles, embeddings = catalog
        rec = _make_recommender(styles, embeddings, "포마드")
        assert len(rec._build_candidate_indices(k=3)) == len(rec.allowed_indices)


# ========== 4. recommend_top_k 통합 ==========


class TestRecommendTopK:
    def test_noisy_style_wins_without_filter(self, catalog):
        """가드 테스트: 필터가 없으면 노이즈 스타일이 top-1을 차지한다"""
        styles, embeddings = catalog
        target = "포마드 (과도한 볼륨)"
        rec = _make_recommender(
            styles, embeddings, target, allowed_indices=list(range(len(styles)))
        )

        results = rec.recommend_top_k(
            k=3, face_features=FACE_FEATURES, skin_features=SKIN_FEATURES
        )
        assert results[0]["hairstyle"] == target

    def test_noisy_style_excluded_with_filter(self, catalog, denylist):
        styles, embeddings = catalog
        target = "포마드 (과도한 볼륨)"
        rec = _make_recommender(styles, embeddings, target)

        results = rec.recommend_top_k(
            k=3, face_features=FACE_FEATURES, skin_features=SKIN_FEATURES
        )
        assert len(results) == 3
        names = [r["hairstyle"] for r in results]
        assert target not in names
        for name in names:
            assert not is_excluded_style(name, denylist), f"노이즈 추천됨: {name}"

    def test_hairstyle_id_matches_npz_index(self, catalog):
        """필터링 후에도 hairstyle_id는 npz 원본 인덱스여야 한다 (피드백 키 호환)"""
        styles, embeddings = catalog
        rec = _make_recommender(styles, embeddings, "포마드 (과도한 볼륨)")

        results = rec.recommend_top_k(
            k=5, face_features=FACE_FEATURES, skin_features=SKIN_FEATURES
        )
        assert results
        for r in results:
            hid = r["hairstyle_id"]
            assert isinstance(hid, int)
            assert 0 <= hid < len(styles)
            assert styles[hid] == r["hairstyle"], (
                f"hairstyle_id 불일치: id={hid} "
                f"npz='{styles[hid]}' vs 추천='{r['hairstyle']}'"
            )

    def test_hairstyle_id_is_not_filtered_position(self, catalog):
        """필터 후 위치 인덱스를 hairstyle_id로 쓰지 않았는지 확인 (회귀 방지)"""
        styles, embeddings = catalog
        rec = _make_recommender(styles, embeddings, "포마드 (과도한 볼륨)")
        allowed = rec._build_candidate_indices(k=3)

        # 제외된 항목이 있으므로 원본 인덱스와 필터 후 위치는 최소 한 곳에서 달라야 한다
        assert any(pos != idx for pos, idx in enumerate(allowed))

        results = rec.recommend_top_k(
            k=3, face_features=FACE_FEATURES, skin_features=SKIN_FEATURES
        )
        for r in results:
            assert r["hairstyle_id"] in allowed

    def test_exclusion_logged_once(self, catalog, caplog):
        styles, embeddings = catalog
        rec = _make_recommender(styles, embeddings, "포마드")

        with caplog.at_level(logging.INFO, logger="models.ml_recommender"):
            rec.recommend_top_k(
                k=3, face_features=FACE_FEATURES, skin_features=SKIN_FEATURES
            )
            rec.recommend_top_k(
                k=3, face_features=FACE_FEATURES, skin_features=SKIN_FEATURES
            )

        hits = [r for r in caplog.records if "[EXCLUDE] 추천 후보 필터링" in r.message]
        assert len(hits) == 1, f"INFO 로그가 1회가 아님: {len(hits)}회"

    def test_gender_filter_still_applies(self, catalog):
        """성별 필터와 제외 필터가 함께 동작한다"""
        styles, embeddings = catalog
        rec = _make_recommender(styles, embeddings, "포마드")
        gender_path = PROJECT_ROOT / "data_source" / "hairstyle_gender.json"
        rec.gender_metadata = json.loads(gender_path.read_text(encoding="utf-8"))

        results = rec.recommend_top_k(
            k=3,
            face_features=FACE_FEATURES,
            skin_features=SKIN_FEATURES,
            gender="male",
        )
        assert results
        for r in results:
            assert rec.gender_metadata.get(r["hairstyle"], "unisex") in (
                "male",
                "unisex",
            )
            assert styles[r["hairstyle_id"]] == r["hairstyle"]
