"""Gemini 이미지 생성 호출 설정 검증

Task 1: 이미지 합성/염색 호출은
- settings.GEMINI_IMAGE_MODEL 모델을 사용하고
- response_modalities 가 ["IMAGE"] (TEXT 미포함) 여야 한다.
"""

import base64
import io
from unittest.mock import MagicMock

import pytest
from PIL import Image

from config.settings import settings
from services.hairstyle_synthesis_service import HairstyleSynthesisService
from services.hair_color_service import HairColorService


def _png_bytes(size=(64, 64), color=(200, 150, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class _InlineData:
    """response.candidates[0].content.parts[i].inline_data 대역"""

    def __init__(self, data: bytes, mime_type: str = "image/png"):
        self.data = data
        self.mime_type = mime_type


class _Part:
    """hasattr(part, "inline_data") 검사를 통과하는 최소 파트 객체"""

    def __init__(self, inline_data=None, text=None):
        self.inline_data = inline_data
        self.text = text


def _image_response(image_bytes: bytes):
    response = MagicMock()
    content = MagicMock()
    content.parts = [_Part(inline_data=_InlineData(image_bytes))]
    candidate = MagicMock()
    candidate.content = content
    response.candidates = [candidate]
    return response


@pytest.fixture
def synthesis_service():
    service = HairstyleSynthesisService()
    client = MagicMock()
    client.models.generate_content.return_value = _image_response(_png_bytes())
    service._client = client
    return service


class TestSynthesisGeminiConfig:
    def test_settings_has_image_model(self):
        """GEMINI_IMAGE_MODEL 설정이 존재하고 셧다운 예정 모델이 아니어야 한다"""
        assert settings.GEMINI_IMAGE_MODEL
        assert settings.GEMINI_IMAGE_MODEL != "gemini-2.5-flash-image"

    def test_class_constant_initialised_from_settings(self):
        """IMAGE_MODEL 클래스 속성은 settings 값에서 초기화된다"""
        assert HairstyleSynthesisService.IMAGE_MODEL == settings.GEMINI_IMAGE_MODEL

    def test_synthesize_hairstyle_uses_image_only_modality(self, synthesis_service):
        result = synthesis_service.synthesize_hairstyle(
            image_data=_png_bytes(), hairstyle_name="투블럭컷", gender="male"
        )

        assert result["success"] is True

        call = synthesis_service._client.models.generate_content.call_args
        assert call.kwargs["model"] == settings.GEMINI_IMAGE_MODEL
        assert call.kwargs["config"].response_modalities == ["IMAGE"]

    def test_synthesize_with_reference_uses_image_only_modality(
        self, synthesis_service
    ):
        result = synthesis_service.synthesize_with_reference(
            user_image_data=_png_bytes(),
            reference_image_data=_png_bytes(size=(48, 48)),
            gender="female",
        )

        assert result["success"] is True

        call = synthesis_service._client.models.generate_content.call_args
        assert call.kwargs["model"] == settings.GEMINI_IMAGE_MODEL
        assert call.kwargs["config"].response_modalities == ["IMAGE"]

    def test_returned_image_is_base64_encoded(self, synthesis_service):
        raw = _png_bytes()
        synthesis_service._client.models.generate_content.return_value = (
            _image_response(raw)
        )

        result = synthesis_service.synthesize_hairstyle(
            image_data=raw, hairstyle_name="레이어드컷"
        )

        assert base64.b64decode(result["image_base64"]) == raw
        assert result["image_format"] == "png"


class TestHairColorGeminiConfig:
    def test_synthesize_hair_color_uses_image_only_modality(self):
        service = HairColorService()
        client = MagicMock()
        client.models.generate_content.return_value = _image_response(_png_bytes())
        service._gemini_client = client

        result = service.synthesize_hair_color(
            image_data=_png_bytes(), color_name="밀크브라운", color_hex="#C4A484"
        )

        assert result["success"] is True

        call = client.models.generate_content.call_args
        assert call.kwargs["model"] == settings.GEMINI_IMAGE_MODEL
        assert call.kwargs["config"].response_modalities == ["IMAGE"]
