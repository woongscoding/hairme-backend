"""스타일 기반 헤어 제품 추천 서비스 (제휴 커머스)

- 카탈로그: data_source/products/catalog.json (사람이 검토/수정하는 시드 파일)
- 제품 수십 개 규모이므로 DynamoDB 대신 JSON 파일로 관리한다 (배포로 갱신,
  Lambda 콜드스타트 시 1회 로드)
- affiliate_url은 추천 목록에 포함하지 않는다 - POST /products/click을 통해서만
  발급해 클릭 로그가 누락되지 않게 한다
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

CATALOG_PATH = (
    Path(__file__).resolve().parents[1] / "data_source" / "products" / "catalog.json"
)

# 추천 목록(공개 응답)에 노출하는 필드 - affiliate_url은 의도적으로 제외
PUBLIC_FIELDS = ("product_id", "name", "brand", "category", "price_krw", "image_url")

# 스타일명 키워드 → category_fallbacks 그룹 (검사 순서 중요: 펌이 컷보다 먼저 -
# "볼륨매직+C컬펌" 같은 복합 스타일은 펌 제품이 우선)
_KEYWORD_GROUPS = (
    (("펌", "웨이브", "물결"), "perm"),
    (("매직",), "magic"),
    (("염색", "컬러", "탈색"), "color"),
    (("컷", "크롭", "페이드", "투블럭", "언더", "올백", "포마드"), "cut"),
)


class ProductRecommendationService:
    """스타일명 → 제휴 제품 추천 (카탈로그 JSON 기반)"""

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

    def get_recommendations(
        self,
        style_name: Optional[str],
        gender: Optional[str] = None,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        스타일에 어울리는 제품 추천 (우선순위 카테고리 순, 최대 limit개)

        모르는 스타일이어도 빈 목록 대신 기본 케어 제품(default 그룹)을 반환한다.
        """
        categories = self._categories_for_style(style_name, gender)

        by_category: Dict[str, List[Dict[str, Any]]] = {}
        for product in self.catalog.get("products", []):
            if product.get("active", True):
                by_category.setdefault(product.get("category", ""), []).append(product)

        picked: List[Dict[str, Any]] = []
        for category in categories:
            for product in by_category.get(category, []):
                picked.append({key: product.get(key) for key in PUBLIC_FIELDS})
                if len(picked) >= limit:
                    return picked
        return picked

    def _categories_for_style(
        self, style_name: Optional[str], gender: Optional[str]
    ) -> List[str]:
        fallbacks = self.catalog.get("category_fallbacks", {})
        default = fallbacks.get("default", [])
        if not style_name:
            return default

        explicit = self.catalog.get("style_mappings", {}).get(style_name)
        if explicit:
            return explicit

        for keywords, group in _KEYWORD_GROUPS:
            if any(keyword in style_name for keyword in keywords):
                # 성별에 따라 제품군 분화: 남성 펌은 왁스 병행, 여성 컷은
                # 왁스/포마드 대신 오일·케어 제품
                if group == "perm" and gender == "male":
                    group = "perm_male"
                elif group == "cut" and gender != "male":
                    group = "cut_female"
                return fallbacks.get(group, default)
        return default


# Singleton
_product_recommendation_service: Optional[ProductRecommendationService] = None


def get_product_recommendation_service() -> ProductRecommendationService:
    global _product_recommendation_service
    if _product_recommendation_service is None:
        _product_recommendation_service = ProductRecommendationService()
    return _product_recommendation_service
