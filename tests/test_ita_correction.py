"""ITA(피부톤 각도) 단위 보정 테스트

OpenCV 8bit LAB 의 L 은 0~255 라서 L* = L * 100/255 로 바꿔야 표준 ITA 가 된다.
- 분류(봄웜/여름쿨/가을웜/겨울쿨)와 퍼스널컬러 밝기는 보정 ITA 를 쓴다.
- 모델 입력(skin_features)·DB 저장(ITA_value)은 레거시 스케일을 그대로 유지한다.
"""

import math
from unittest.mock import MagicMock

import pytest

from models.mediapipe_analyzer import (
    MediaPipeFaceFeatures,
    classify_skin_tone,
    compute_ita,
)


def _old_formula(L_cv, b):
    """수정 전 코드의 ITA 공식 (L_cv 를 그대로 L* 로 사용)"""
    if abs(b) > 0.1:
        return math.atan((L_cv - 50) / b) * 180 / math.pi
    return 0.0


class TestComputeIta:
    def test_corrected_ita_uses_L_star(self):
        # L_cv 166 → L* ≈ 65.1, b* 15 → ITA ≈ 45°
        _, corrected = compute_ita(166.0, 15.0)
        assert corrected == pytest.approx(45.2, abs=0.5)

    @pytest.mark.parametrize(
        "L_cv,b", [(166.0, 15.0), (128.0, 14.0), (190.0, 8.0), (100.0, 20.0)]
    )
    def test_legacy_ita_unchanged(self, L_cv, b):
        """모델 입력용 레거시 값은 예전 공식과 비트 단위로 같아야 한다"""
        legacy, _ = compute_ita(L_cv, b)
        assert legacy == _old_formula(L_cv, b)

    def test_legacy_is_always_large_for_skin(self):
        """버그 재현: 레거시 값은 일반 피부에서 항상 42° 를 넘는다"""
        legacy, corrected = compute_ita(128.0, 14.0)  # L* ≈ 50 (어두운 편)
        assert legacy > 42
        assert corrected < 10

    def test_near_zero_b_returns_zero(self):
        assert compute_ita(170.0, 0.05) == (0.0, 0.0)


class TestClassifySkinTone:
    def test_light_warm_is_spring(self):
        _, ita = compute_ita(180.0, 14.0)  # L* ≈ 70.6 → ITA ≈ 55.8
        assert classify_skin_tone(ita, 11.0, 14.0) == "봄웜"

    def test_light_neutral_is_summer(self):
        _, ita = compute_ita(180.0, 4.0)
        assert classify_skin_tone(ita, 12.0, 4.0) == "여름쿨"

    def test_medium_warm_reaches_autumn(self):
        _, ita = compute_ita(155.0, 15.0)  # L* ≈ 60.8 → ITA ≈ 35.7
        assert 32 < ita <= 42
        assert classify_skin_tone(ita, 11.0, 15.0) == "가을웜"

    def test_dark_warm_reaches_autumn(self):
        _, ita = compute_ita(135.0, 16.0)  # L* ≈ 52.9 → ITA ≈ 10.3
        assert ita <= 22
        assert classify_skin_tone(ita, 12.0, 16.0) == "가을웜"

    def test_dark_cool_reaches_winter(self):
        _, ita = compute_ita(135.0, 12.0)
        assert ita <= 22
        assert classify_skin_tone(ita, 19.0, 12.0) == "겨울쿨"

    def test_legacy_scale_could_never_reach_autumn_or_winter(self):
        """같은 어두운 입력이라도 레거시 값으로 분류하면 봄웜/여름쿨만 나온다 (버그)"""
        legacy, _ = compute_ita(135.0, 12.0)
        assert classify_skin_tone(legacy, 19.0, 12.0) == "여름쿨"
        assert classify_skin_tone(legacy, 12.0, 16.0) == "봄웜"


class TestFeaturesDataclass:
    def _features(self, **kw):
        base = dict(
            face_shape="계란형",
            skin_tone="봄웜",
            confidence=0.9,
            face_ratio=1.2,
            forehead_width=100.0,
            cheekbone_width=120.0,
            jaw_width=100.0,
            ITA_value=83.0,
            hue_value=11.0,
            face_features=[1.2, 100.0, 120.0, 100.0, 0.8, 0.8],
            skin_features=[83.0, 11.0],
        )
        base.update(kw)
        return MediaPipeFaceFeatures(**base)

    def test_ita_corrected_defaults_to_none(self):
        f = self._features()
        assert f.ITA_corrected is None
        assert f.to_dict()["ITA_corrected"] is None

    def test_to_dict_keeps_legacy_fields(self):
        f = self._features(ITA_corrected=45.2)
        d = f.to_dict()
        assert d["ITA_value"] == 83.0
        assert d["skin_features"] == [83.0, 11.0]
        assert d["ITA_corrected"] == 45.2


class TestPersonalColorBrightness:
    def _service(self, features):
        from services.personal_color_service import PersonalColorService

        svc = PersonalColorService()
        svc._analyzer = MagicMock()
        svc._analyzer.analyze.return_value = features
        svc._personal_color_data = {}
        svc._hair_color_data = {}
        return svc

    def _features(self, legacy, corrected):
        f = MagicMock()
        f.skin_tone = "가을웜"
        f.confidence = 0.9
        f.ITA_value = legacy
        f.ITA_corrected = corrected
        f.hue_value = 11.0
        return f

    def test_brightness_uses_corrected_ita(self):
        svc = self._service(self._features(legacy=83.0, corrected=35.0))
        result = svc.analyze(b"img")
        assert result is not None
        assert result.brightness == "medium"
        assert result.ita_value == 35.0

    def test_dark_skin_is_muted(self):
        svc = self._service(self._features(legacy=81.0, corrected=10.0))
        assert svc.analyze(b"img").brightness == "muted"

    def test_falls_back_to_legacy_when_corrected_missing(self):
        svc = self._service(self._features(legacy=83.0, corrected=None))
        result = svc.analyze(b"img")
        assert result.brightness == "bright"
        assert result.ita_value == 83.0
