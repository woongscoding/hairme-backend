"""헤어스타일 예시 이미지 URL 해석 서비스

data_source/style_images/styles.json(스타일→image_key 매핑)과
image_manifest.json(image_key→WebP 파일)을 로드해서
추천 응답의 각 스타일에 대한 예시 이미지 URL을 결정한다.

이미지는 Lambda 이미지에 번들된 WebP를 GET /api/styles/images/{filename}으로
서빙한다 (S3 권한 없이 동작, 추후 CDN 도입 시 이 서비스의 URL 생성만 교체).

매핑 파일이 없으면 모든 조회가 None을 반환한다 (추천 자체는 정상 동작).
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import quote

from core.logging import logger
from utils.style_preprocessor import normalize_style_name

_STYLE_DIR = Path(__file__).parent.parent / "data_source" / "style_images"
_STYLES_PATH = _STYLE_DIR / "styles.json"
_MANIFEST_PATH = _STYLE_DIR / "image_manifest.json"

IMAGE_ENDPOINT_PREFIX = "/api/styles/images/"


def _canonical_name(raw: str) -> str:
    """scripts/extract_style_catalog.py의 canonical_name과 동일 규칙"""
    name = re.sub(r"\([^)]*\)", "", raw)
    name = name.split(": ")[0]
    name = name.strip().rstrip(":").strip()
    return name.replace(" ", "")


class StyleImageService:
    """스타일명/ID → 예시 이미지 URL 해석"""

    def __init__(self):
        self._id_to_key: Dict[int, str] = {}
        self._name_to_key: Dict[str, str] = {}
        self._manifest: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(_STYLES_PATH, encoding="utf-8") as f:
                styles = json.load(f)
            with open(_MANIFEST_PATH, encoding="utf-8") as f:
                self._manifest = json.load(f)
        except FileNotFoundError as e:
            logger.warning(f"스타일 이미지 매핑 파일 없음 - image_url 비활성화: {e}")
            return
        except Exception as e:
            logger.error(f"스타일 이미지 매핑 로드 실패 - image_url 비활성화: {e}")
            return

        for item in styles.get("ml_styles", []):
            key = item["image_key"]
            self._id_to_key[item["hairstyle_id"]] = key
            self._name_to_key[item["raw_name"]] = key
            self._name_to_key[normalize_style_name(item["raw_name"])] = key
            self._name_to_key[item["canonical"]] = key

        for item in styles.get("trending_styles", []):
            key = item["image_key"]
            self._name_to_key[item["style_name"]] = key
            self._name_to_key[normalize_style_name(item["style_name"])] = key
            self._name_to_key[item["canonical"]] = key

        logger.info(
            f"✅ 스타일 이미지 매핑 로드: 스타일 {len(self._name_to_key)}개 항목, "
            f"이미지 {len(self._manifest)}개"
        )

    @property
    def enabled(self) -> bool:
        return bool(self._manifest)

    def _resolve_key(
        self, style_name: Optional[str], hairstyle_id: Optional[int]
    ) -> Optional[str]:
        if hairstyle_id is not None:
            try:
                key = self._id_to_key.get(int(hairstyle_id))
                if key:
                    return key
            except (TypeError, ValueError):
                pass

        if not style_name:
            return None

        for candidate in (
            style_name,
            normalize_style_name(style_name),
            _canonical_name(style_name),
        ):
            key = self._name_to_key.get(candidate)
            if key:
                return key
        return None

    def resolve_filename(
        self,
        style_name: Optional[str] = None,
        hairstyle_id: Optional[int] = None,
        gender: Optional[str] = None,
    ) -> Optional[str]:
        """스타일에 해당하는 WebP 파일명 반환 (없으면 None)

        gender와 일치하는 이미지를 우선하고, 없으면 반대 성별 이미지로 폴백
        (예: 여성 사용자에게 남성 전용 스타일이 추천된 경우).
        """
        key = self._resolve_key(style_name, hairstyle_id)
        if not key:
            return None

        gender_order = ["female", "male"] if gender == "female" else ["male", "female"]
        for g in gender_order:
            entry = self._manifest.get(f"{g}/{key}")
            if entry:
                return entry["file"]
        return None

    def build_image_url(
        self,
        style_name: Optional[str] = None,
        hairstyle_id: Optional[int] = None,
        gender: Optional[str] = None,
        base_url: str = "",
    ) -> Optional[str]:
        """예시 이미지의 절대(또는 상대) URL 반환 (없으면 None)

        Args:
            base_url: 요청의 base URL (예: str(request.base_url)).
                      비어있으면 상대 경로를 반환한다.
        """
        filename = self.resolve_filename(style_name, hairstyle_id, gender)
        if not filename:
            return None

        path = IMAGE_ENDPOINT_PREFIX + quote(filename)
        if base_url:
            return base_url.rstrip("/") + path
        return path


# ========== 싱글톤 인스턴스 ==========
_style_image_service: Optional[StyleImageService] = None


def get_style_image_service() -> StyleImageService:
    global _style_image_service
    if _style_image_service is None:
        _style_image_service = StyleImageService()
    return _style_image_service
