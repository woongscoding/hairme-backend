"""무료 등급 합성 결과 워터마크

우하단에 반투명 앱 이름을 얹는다. 실패하면 원본을 그대로 돌려준다
(워터마크 때문에 합성 응답이 깨지면 안 된다 - 이미 Gemini 비용은 나갔다).
"""

import base64
import io
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

from core.logging import logger

# 짧은 변 대비 글자 높이 비율 (512px 결과에서 약 28px)
_FONT_RATIO = 0.055
_MARGIN_RATIO = 0.03


def apply_watermark(image_base64: str, image_format: str, text: str) -> Tuple[str, str]:
    """
    Returns:
        (워터마크가 들어간 base64, 포맷). 실패 시 입력 그대로.
    """
    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
        base = image.convert("RGBA")
        width, height = base.size
        font_size = max(12, int(min(width, height) * _FONT_RATIO))
        font = ImageFont.load_default(size=font_size)

        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        margin = int(min(width, height) * _MARGIN_RATIO)
        x = width - (right - left) - margin - left
        y = height - (bottom - top) - margin - top
        # 밝은 배경에서도 보이도록 그림자 1px + 흰 글자
        draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, 110))
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 190))

        composed = Image.alpha_composite(base, overlay)
        buffer = io.BytesIO()
        fmt = (image_format or "png").lower()
        if fmt in ("jpeg", "jpg"):
            composed.convert("RGB").save(buffer, format="JPEG", quality=90)
            out_format = "jpeg"
        else:
            composed.save(buffer, format="PNG")
            out_format = "png"
        return base64.b64encode(buffer.getvalue()).decode("utf-8"), out_format
    except Exception as e:
        logger.warning(f"워터마크 적용 실패 (원본 반환): {str(e)}")
        return image_base64, image_format
