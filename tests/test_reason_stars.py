# -*- coding: utf-8 -*-
"""별점 문구 숨김 테스트 (작업 2)

재학습 모델(v6_feedback_20260909)이 40~65 점대를 내는데 별점 기준은
75/80/85/90 고정이라 모든 추천 카드가 "★☆☆ 추천" 으로 나간다.
기준은 그대로 두고 문구만 감춘다(settings.SHOW_CONFIDENCE_STARS, 기본 False).

- 기본값에서는 사유에 별점 접두어가 없다
- 플래그를 올리면 기존 문구가 그대로 되돌아온다 (로직 보존)
- 구버전 앱이 사유 문자열을 그대로 표시하므로 문장이 어색하게 잘리지 않는다
- 응답 스키마의 필드(이름/타입)는 바뀌지 않는다
"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")

import pytest

from config.settings import settings
from services.reason_generator import ReasonGenerator

# 재학습 모델의 실측 점수대 + 옛 기준의 경계값
CURRENT_MODEL_SCORES = [40.0, 47.5, 52.3, 58.0, 65.0]
LEGACY_BAND_SCORES = [95.0, 90.0, 87.0, 85.0, 82.0, 80.0, 77.0, 75.0, 70.0]

STARS = ("★", "☆")


@pytest.fixture
def generator():
    return ReasonGenerator()


@pytest.fixture
def stars_hidden(monkeypatch):
    monkeypatch.setattr(settings, "SHOW_CONFIDENCE_STARS", False)


@pytest.fixture
def stars_shown(monkeypatch):
    monkeypatch.setattr(settings, "SHOW_CONFIDENCE_STARS", True)


class TestStarsHiddenByDefault:
    def test_default_setting_is_off(self):
        """기본값은 꺼짐 - 배포 즉시 문구가 사라진다"""
        assert settings.SHOW_CONFIDENCE_STARS is False

    @pytest.mark.parametrize("score", CURRENT_MODEL_SCORES + LEGACY_BAND_SCORES)
    def test_no_star_characters_at_any_score(self, generator, stars_hidden, score):
        reason = generator.generate_with_score("계란형", "봄웜", "레이어드 컷", score)

        assert not any(star in reason for star in STARS)

    def test_reason_is_exactly_the_base_template(self, generator, stars_hidden):
        """접두어만 빠지고 템플릿 문장은 그대로여야 한다"""
        reason = generator.generate_with_score("둥근형", "봄웜", "허쉬컷", 52.0)

        expected = [
            template.format(style="허쉬컷")
            for template in ReasonGenerator.FACE_TEMPLATES["둥근형"]
        ]
        assert reason in expected

    @pytest.mark.parametrize("score", CURRENT_MODEL_SCORES)
    def test_sentence_is_not_awkwardly_truncated(self, generator, stars_hidden, score):
        """구버전 앱은 이 문자열을 그대로 표시한다 - 꼬리가 잘리면 안 된다"""
        reason = generator.generate_with_score("각진형", "겨울쿨", "레이어드 컷", score)

        assert reason == reason.strip()  # 앞뒤 공백 없음
        assert reason  # 빈 문자열 아님
        assert not reason.endswith((",", "(", "-", "및", "그리고"))
        # 한국어 종결어미로 끝난다 (템플릿은 모두 "~합니다/~줍니다" 형태)
        assert reason.endswith("다")
        assert "  " not in reason  # 접두어가 빠진 자리에 빈칸이 남지 않음

    def test_all_face_shapes_and_styles_stay_clean(self, generator, stars_hidden):
        for face_shape in ReasonGenerator.FACE_TEMPLATES:
            for _ in range(20):  # 템플릿은 무작위 선택이므로 반복
                reason = generator.generate_with_score(
                    face_shape, "봄웜", "시스루뱅", 48.0
                )
                assert not any(star in reason for star in STARS)
                assert reason == reason.strip()
                assert reason.endswith("다")


class TestRollback:
    """플래그를 올리면 기존 로직이 그대로 살아있다"""

    @pytest.mark.parametrize(
        "score,expected",
        [
            (95.0, "★★★ 강력 추천"),
            (90.0, "★★★ 강력 추천"),
            (87.0, "★★★ 매우 잘 어울림"),
            (85.0, "★★★ 매우 잘 어울림"),
            (82.0, "★★☆ 추천"),
            (80.0, "★★☆ 추천"),
            (77.0, "★★☆ 잘 어울림"),
            (75.0, "★★☆ 잘 어울림"),
            (70.0, "★☆☆ 추천"),
            (52.0, "★☆☆ 추천"),
        ],
    )
    def test_thresholds_unchanged(self, generator, stars_shown, score, expected):
        reason = generator.generate_with_score("계란형", "봄웜", "레이어드 컷", score)

        assert reason.endswith(f" {expected}")

    def test_base_reason_still_prefixes_the_confidence(self, generator, stars_shown):
        reason = generator.generate_with_score("긴형", "여름쿨", "단발 보브", 91.0)

        base, _, confidence = reason.rpartition(" ★★★ ")
        assert confidence == "강력 추천"
        assert base in [
            template.format(style="단발 보브")
            for template in ReasonGenerator.FACE_TEMPLATES["긴형"]
        ]


class TestResponseSchemaUnchanged:
    """사유는 여전히 문자열 - 추천 카드의 필드 이름/타입은 그대로"""

    def _recommendation(self, monkeypatch, show_stars):
        from services.hybrid_recommender import MLRecommendationService

        monkeypatch.setattr(settings, "SHOW_CONFIDENCE_STARS", show_stars)

        service = MLRecommendationService.__new__(MLRecommendationService)
        service.reason_generator = ReasonGenerator()

        built = service._build_recommendations(
            [{"hairstyle_id": 7, "hairstyle": "레이어드 컷", "score": 52.31}],
            "계란형",
            "봄웜",
        )
        return built[0]

    def test_fields_and_types_identical_in_both_modes(self, monkeypatch):
        hidden = self._recommendation(monkeypatch, False)
        shown = self._recommendation(monkeypatch, True)

        assert set(hidden) == set(shown)
        assert set(hidden) == {
            "hairstyle_id",
            "style_name",
            "reason",
            "source",
            "score",
            "rank",
        }
        for key in hidden:
            assert type(hidden[key]) is type(shown[key])

        # score 는 기존대로 0-1 범위 (별점 숨김과 무관)
        assert hidden["score"] == 0.52
        assert isinstance(hidden["reason"], str)
        assert not any(star in hidden["reason"] for star in STARS)
        assert any(star in shown["reason"] for star in STARS)
