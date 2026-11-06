"""합성 결과 화질 등급 (수익화 v2.1)

차감 결과(quota["tier"])로 등급을 정하고, 등급별로 출력 해상도·워터마크를 고른다.
스타일 합성(synthesis.py)과 염색 합성(hair_color.py)이 같은 규칙을 쓴다.

| 등급 | 해상도           | 워터마크              | is_preview |
| ---- | ---------------- | --------------------- | ---------- |
| free | FREE_IMAGE_SIZE  | FREE_RESULT_WATERMARK | true       |

모델이 FREE_IMAGE_SIZE 보다 크게 돌려주면(Nano Banana 는 512 미지원 → 1K) 워터마크 전에
긴 변을 그 크기로 줄인다. 무료 결과의 화질 차이는 모델과 상관없이 유지된다.
| paid | PAID_IMAGE_SIZE  | 없음                  | false      |
"""

import base64
import io
from typing import Optional, Tuple

from PIL import Image

from config.settings import settings
from core.logging import logger
from services.credit_service import TIER_FREE, TIER_PAID
from services.result_watermark import apply_watermark

# image_size 설정값 → 긴 변 픽셀 (줄일 수 있는 것만)
_SIZE_PIXELS = {"512": 512}


def charged_tier(quota: Optional[dict]) -> str:
    """차감 결과의 등급 (quota 가 없거나 등급이 비면 무료)"""
    return (quota or {}).get("tier") or TIER_FREE


def result_quality(tier: str) -> dict:
    """응답의 quality 필드 (앱이 "고화질로 보기" 버튼 노출을 판단)"""
    paid = tier == TIER_PAID
    return {
        "tier": tier,
        "image_size": (settings.PAID_IMAGE_SIZE if paid else settings.FREE_IMAGE_SIZE)
        or None,
        "watermarked": (not paid) and settings.FREE_RESULT_WATERMARK,
    }


def is_preview(tier: str) -> bool:
    """무료 미리보기 결과인지 (앱 계약 필드 - true 면 고화질 결제 유도)"""
    return tier != TIER_PAID


def image_size_for(tier: str) -> Optional[str]:
    """Gemini 에 넘길 출력 해상도 (설정이 비면 None = 모델 기본값)"""
    size = settings.PAID_IMAGE_SIZE if tier == TIER_PAID else settings.FREE_IMAGE_SIZE
    return size or None


def downscale_to(
    image_base64: str, image_format: str, size: Optional[str]
) -> Tuple[str, str]:
    """긴 변이 size 보다 크면 그 크기로 줄인다. 실패하거나 해당 없으면 그대로."""
    max_side = _SIZE_PIXELS.get(size or "")
    if not max_side:
        return image_base64, image_format
    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
        if max(image.size) <= max_side:
            return image_base64, image_format
        scale = max_side / max(image.size)
        resized = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.LANCZOS
        )
        buffer = io.BytesIO()
        fmt = (image_format or "png").lower()
        if fmt in ("jpeg", "jpg"):
            resized.convert("RGB").save(buffer, format="JPEG", quality=90)
        else:
            resized.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8"), image_format
    except Exception as e:
        logger.warning(f"무료 결과 축소 실패 (원본 반환): {str(e)}")
        return image_base64, image_format


def finalize_image(result: dict, tier: str) -> tuple:
    """무료 등급이면 FREE_IMAGE_SIZE 로 줄이고 워터마크, 유료면 그대로"""
    image_base64, image_format = result["image_base64"], result["image_format"]
    if tier == TIER_PAID:
        return image_base64, image_format
    image_base64, image_format = downscale_to(
        image_base64, image_format, settings.FREE_IMAGE_SIZE
    )
    if not result_quality(tier)["watermarked"]:
        return image_base64, image_format
    return apply_watermark(image_base64, image_format, settings.RESULT_WATERMARK_TEXT)
