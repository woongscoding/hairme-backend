"""Gemini 이미지 생성(헤어 합성·염색 합성) 요청 설정

모델마다 지원하는 옵션이 달라서 한곳에서 맞춘다.

- Nano Banana 2.1 (gemini-nano-banana-*)
  * 출력 512 를 지원하지 않는다 → 1K 로 받고, 무료 등급은 finalize 단계에서 512 로 줄인다
  * 기본 생각 수준이 medium 이라 느리다(최대 27초 측정) → GEMINI_IMAGE_THINKING_LEVEL(minimal)
- gemini-3.1-flash-image 등 이전 모델
  * 512 를 그대로 요청하고 생각 설정은 넣지 않는다 (지원하지 않을 수 있음)

그래서 GEMINI_IMAGE_MODEL 하나만 이전 모델로 바꿔도 그대로 롤백된다.
"""

from typing import Any, Dict, Optional

from config.settings import settings

# 출력 512 를 지원하지 않고 생각 수준을 고를 수 있는 모델 계열
_NANO_BANANA_PREFIX = "gemini-nano-banana"


def is_nano_banana(model: str) -> bool:
    return model.startswith(_NANO_BANANA_PREFIX)


def request_image_size(image_size: Optional[str], model: str) -> Optional[str]:
    """모델에 실제로 요청할 해상도. 512 미지원 모델은 1K 로 받는다."""
    if image_size == "512" and is_nano_banana(model):
        return "1K"
    return image_size


def image_generation_config(
    types, image_size: Optional[str], model: Optional[str] = None
):
    """
    Args:
        types: google.genai.types (호출 측에서 지연 import 한 모듈)
        image_size: 원하는 출력 해상도 ("512" | "1K" | "2K" | None=모델 기본값)
        model: 기본은 settings.GEMINI_IMAGE_MODEL
    """
    model = model or settings.GEMINI_IMAGE_MODEL
    kwargs: Dict[str, Any] = {
        # IMAGE 단독 요청이 TEXT 병행 대비 약 2.6초 빠르고,
        # "텍스트만 반환 → 재시도" 실패 모드를 제거한다.
        "response_modalities": ["IMAGE"],
    }
    size = request_image_size(image_size, model)
    if size:
        kwargs["image_config"] = types.ImageConfig(image_size=size)
    if is_nano_banana(model) and settings.GEMINI_IMAGE_THINKING_LEVEL:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=settings.GEMINI_IMAGE_THINKING_LEVEL
        )
    return types.GenerateContentConfig(**kwargs)
