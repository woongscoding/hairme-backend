"""얼굴형 분류 임계값 테스트 (models/mediapipe_analyzer.py classify_from_ratios).

2026-09-17 재보정: 운영 측정값 분포(face_ratio 1.228±0.061, jaw_ratio 0.783±0.025,
forehead_ratio 0.845±0.026)의 분위수를 임계값으로 쓴다. 이 테스트는
(1) 다섯 얼굴형이 모두 도달 가능하고 (2) 분포의 중심이 계란형이며
(3) 신뢰도가 경계에서 멀어질수록 단조 증가함을 고정한다.
"""

import sys
import types

import pytest


@pytest.fixture(scope="module")
def analyzer_cls():
    """mediapipe 없이 클래스만 가져온다 (분류 함수는 순수 함수)."""
    for name in ("mediapipe", "mediapipe.solutions", "mediapipe.solutions.face_mesh"):
        sys.modules.setdefault(name, types.ModuleType(name))
    from models.mediapipe_analyzer import MediaPipeFaceAnalyzer

    return MediaPipeFaceAnalyzer


# 운영 분포의 평균값 — 계란형이어야 한다
MEAN = dict(face_ratio=1.228, forehead_ratio=0.845, jaw_ratio=0.783)


def test_population_mean_is_oval(analyzer_cls):
    shape, conf = analyzer_cls.classify_from_ratios(**MEAN)
    assert shape == "계란형"
    assert 0.6 <= conf <= 0.9


@pytest.mark.parametrize(
    "face_ratio,forehead_ratio,jaw_ratio,expected",
    [
        (1.31, 0.845, 0.783, "긴형"),  # p95 세로 비율
        (1.13, 0.845, 0.783, "둥근형"),  # p5 세로 비율
        (1.228, 0.845, 0.82, "각진형"),  # p95 턱 비율
        (1.228, 0.80, 0.74, "하트형"),  # p5 이마·턱 모두 좁음
        (1.228, 0.80, 0.783, "계란형"),  # 이마만 좁으면 하트형 아님
        (1.228, 0.845, 0.74, "계란형"),  # 턱만 좁으면 하트형 아님
    ],
)
def test_each_shape_is_reachable(
    analyzer_cls, face_ratio, forehead_ratio, jaw_ratio, expected
):
    shape, _ = analyzer_cls.classify_from_ratios(face_ratio, forehead_ratio, jaw_ratio)
    assert shape == expected


def test_face_ratio_takes_priority_over_jaw(analyzer_cls):
    # 세로 비율이 긴형 구간이면 턱이 넓어도 긴형
    shape, _ = analyzer_cls.classify_from_ratios(1.35, 0.845, 0.85)
    assert shape == "긴형"


def test_confidence_is_monotone_and_bounded(analyzer_cls):
    cls = analyzer_cls
    prev = 0.0
    for fr in (1.28, 1.30, 1.33, 1.38, 1.50):
        shape, conf = cls.classify_from_ratios(fr, 0.845, 0.783)
        assert shape == "긴형"
        assert 0.6 <= conf <= 0.9
        assert conf >= prev
        prev = conf
    assert prev == 0.9  # 경계에서 충분히 멀면 상한


def test_old_thresholds_no_longer_swallow_everything(analyzer_cls):
    """이전 규칙에서 100% 계란형이던 p5~p95 격자가 다섯 유형으로 갈라져야 한다."""
    cls = analyzer_cls
    shapes = set()
    for fr in (1.13, 1.20, 1.24, 1.27, 1.31):
        for fo in (0.80, 0.845, 0.89):
            for jr in (0.74, 0.783, 0.82):
                shapes.add(cls.classify_from_ratios(fr, fo, jr)[0])
    assert shapes == {"계란형", "둥근형", "긴형", "각진형", "하트형"}
