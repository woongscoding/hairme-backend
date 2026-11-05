"""체형별 아이템 추천 서비스 (제휴 커머스)

- 카탈로그: data_source/fashion/body_catalog.json (사람이 검토/수정하는 시드 파일)
- 헤어 제품 카탈로그와 같은 방식: JSON 파일, 콜드스타트 후 첫 사용 시 1회 로드
- affiliate_url은 추천 응답에 넣지 않는다 - POST /products/click으로만 발급해
  클릭 로그가 누락되지 않게 한다
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

CATALOG_PATH = (
    Path(__file__).resolve().parents[1]
    / "data_source"
    / "fashion"
    / "body_catalog.json"
)

# 추천 응답에 노출하는 필드 - affiliate_url은 의도적으로 제외
PUBLIC_FIELDS = (
    "product_id",
    "name",
    "brand",
    "category",
    "fit_keyword",
    "price_krw",
    "image_url",
)


class BodyFashionService:
    """체형 분류 결과 → 추천 핏/피할 핏/제휴 아이템"""

    def __init__(self, catalog_path: Path = CATALOG_PATH):
        self._catalog_path = catalog_path
        self._catalog: Optional[Dict[str, Any]] = None

    @property
    def catalog(self) -> Dict[str, Any]:
        if self._catalog is None:
            with open(self._catalog_path, encoding="utf-8") as f:
                self._catalog = json.load(f)
        return self._catalog

    @property
    def disclosure(self) -> str:
        """공정위 표시광고 의무 문구 - 추천이 노출되는 모든 응답에 포함할 것"""
        return self.catalog.get("disclosure", "")

    def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """product_id로 제품 조회 (affiliate_url 포함 전체 필드, 비활성 제품 제외)"""
        for product in self.catalog.get("products", []):
            if product.get("product_id") == product_id and product.get("active", True):
                return product
        return None

    def build_recommendation(
        self, body_type: str, flags: Sequence[str], gender: str
    ) -> Dict[str, Any]:
        """체형 + 보조 표시 + 성별로 응답 본문을 만든다.

        보조 표시(예: 상체 긴 비율)의 추천은 체형 추천 뒤에 붙이고,
        같은 키워드는 한 번만 넣는다.
        """
        key = "female" if gender == "female" else "male"
        entry = self.catalog["body_types"][body_type]
        sections = [entry[key]]

        secondary: List[Dict[str, str]] = []
        for flag in flags:
            proportion = self.catalog.get("proportions", {}).get(flag)
            if not proportion:
                continue
            secondary.append(
                {
                    "key": flag,
                    "label": proportion["label"],
                    "description": proportion["description"],
                }
            )
            sections.append(proportion[key])

        return {
            "body_type": body_type,
            "body_type_label": entry["label"],
            "description": entry["description"],
            "secondary": secondary,
            "recommended": _merge(sections, "recommended"),
            "avoid": _merge(sections, "avoid"),
            "items": self._items(sections),
            "disclosure": self.disclosure,
        }

    def _items(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        seen = set()
        for section in sections:
            for product_id in section.get("items", []):
                if product_id in seen:
                    continue
                seen.add(product_id)
                product = self.get_product(product_id)
                if product is not None:
                    items.append({k: product.get(k) for k in PUBLIC_FIELDS})
        return items


def _merge(sections: List[Dict[str, Any]], field: str) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []
    seen = set()
    for section in sections:
        for fit in section.get(field, []):
            if fit["keyword"] not in seen:
                seen.add(fit["keyword"])
                merged.append({"keyword": fit["keyword"], "reason": fit["reason"]})
    return merged


# Singleton
_body_fashion_service: Optional[BodyFashionService] = None


def get_body_fashion_service() -> BodyFashionService:
    global _body_fashion_service
    if _body_fashion_service is None:
        _body_fashion_service = BodyFashionService()
    return _body_fashion_service
