"""이미지 모델 전환 (Nano Banana 2.1): 모델별 요청 설정 + 무료 결과 축소"""

import base64
import io

import pytest
from google.genai import types
from PIL import Image

from config.settings import settings
from services.gemini_image_config import image_generation_config, request_image_size
from services.result_quality import downscale_to, finalize_image


def _png_b64(width: int, height: int) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _size(image_base64: str):
    return Image.open(io.BytesIO(base64.b64decode(image_base64))).size


class TestRequestConfig:
    def test_default_model_is_nano_banana_with_minimal_thinking(self):
        assert settings.GEMINI_IMAGE_MODEL == "gemini-nano-banana-2.1"
        config = image_generation_config(types, "1K")
        assert config.response_modalities == ["IMAGE"]
        assert config.image_config.image_size == "1K"
        assert config.thinking_config.thinking_level.lower() == "minimal"

    def test_nano_banana_gets_1k_instead_of_512(self):
        assert request_image_size("512", "gemini-nano-banana-2.1") == "1K"
        config = image_generation_config(types, "512", model="gemini-nano-banana-2.1")
        assert config.image_config.image_size == "1K"

    def test_rollback_model_keeps_512_and_no_thinking(self):
        config = image_generation_config(types, "512", model="gemini-3.1-flash-image")
        assert config.image_config.image_size == "512"
        assert config.thinking_config is None

    def test_no_size_leaves_model_default(self):
        config = image_generation_config(types, None)
        assert config.image_config is None

    def test_empty_thinking_level_uses_model_default(self, monkeypatch):
        monkeypatch.setattr(settings, "GEMINI_IMAGE_THINKING_LEVEL", "")
        assert image_generation_config(types, "1K").thinking_config is None


class TestFreeResultDownscale:
    def test_1k_result_is_reduced_to_512_long_side(self):
        out, fmt = downscale_to(_png_b64(895, 1200), "png", "512")
        assert _size(out) == (382, 512)
        assert fmt == "png"

    def test_small_result_is_untouched(self):
        original = _png_b64(384, 512)
        assert downscale_to(original, "png", "512") == (original, "png")

    def test_non_pixel_size_is_untouched(self):
        original = _png_b64(895, 1200)
        assert downscale_to(original, "png", "1K") == (original, "png")

    def test_broken_image_returns_original(self):
        assert downscale_to("not-base64", "png", "512") == ("not-base64", "png")

    def test_free_tier_is_reduced_then_watermarked(self):
        result = {"image_base64": _png_b64(895, 1200), "image_format": "png"}
        out, _ = finalize_image(result, "free")
        assert max(_size(out)) == 512

    @pytest.mark.parametrize("tier", ["paid"])
    def test_paid_tier_keeps_full_size(self, tier):
        result = {"image_base64": _png_b64(895, 1200), "image_format": "png"}
        out, _ = finalize_image(result, tier)
        assert _size(out) == (895, 1200)
