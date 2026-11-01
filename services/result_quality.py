"""합성 결과 화질 등급 (수익화 v2.1)

차감 결과(quota["tier"])로 등급을 정하고, 등급별로 출력 해상도·워터마크를 고른다.
스타일 합성(synthesis.py)과 염색 합성(hair_color.py)이 같은 규칙을 쓴다.

| 등급 | 해상도           | 워터마크              | is_preview |
| ---- | ---------------- | --------------------- | ---------- |
| free | FREE_IMAGE_SIZE  | FREE_RESULT_WATERMARK | true       |
| paid | PAID_IMAGE_SIZE  | 없음                  | false      |
"""

from typing import Optional

from config.settings import settings
from services.credit_service import TIER_FREE, TIER_PAID
from services.result_watermark import apply_watermark


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


def finalize_image(result: dict, tier: str) -> tuple:
    """무료 등급이면 워터마크를 넣은 (base64, format), 유료면 그대로"""
    if not result_quality(tier)["watermarked"]:
        return result["image_base64"], result["image_format"]
    return apply_watermark(
        result["image_base64"], result["image_format"], settings.RESULT_WATERMARK_TEXT
    )
