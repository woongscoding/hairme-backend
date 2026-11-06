"""체형별 아이템 추천 서비스 (제휴 커머스)

- 카탈로그: data_source/fashion/body_catalog.json (사람이 검토/수정하는 시드 파일)
- 헤어 제품 카탈로그와 같은 방식: JSON 파일, 콜드스타트 후 첫 사용 시 1회 로드
- 링크는 상품이 아니라 쿠팡 검색 결과 페이지다. 쿠팡 오픈 API는 파트너스 최종 승인
  전에는 쓸 수 없어서, 추천 키워드 x 성별마다 검색 링크 하나를 둔다
  (교체 절차: scripts/fashion_links.py)
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


class BodyFashionService:
    """체형 분류 결과 → 추천 핏(검색 링크 ID 포함)/피할 핏"""

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
        """product_id로 검색 링크 조회 (affiliate_url 포함, 비활성 제외)

        클릭 엔드포인트가 헤어 제품과 같은 이름으로 부르므로 get_product 로 둔다.
        """
        for link in self.catalog.get("links", []):
            if link.get("product_id") == product_id and link.get("active", False):
                return link
        return None

    def _active_link_ids(self, gender: str) -> Dict[str, str]:
        """keyword → product_id (해당 성별의 활성 링크만)"""
        return {
            link["keyword"]: link["product_id"]
            for link in self.catalog.get("links", [])
            if link.get("gender") == gender and link.get("active", False)
        }

    def build_recommendation(
        self, body_type: str, flags: Sequence[str], gender: str
    ) -> Dict[str, Any]:
        """체형 + 보조 표시 + 성별로 응답 본문을 만든다.

        보조 표시(예: 상체 긴 비율)의 추천은 체형 추천 뒤에 붙이고,
        같은 키워드는 한 번만 넣는다. 추천 키워드마다 product_id 를 붙이는데,
        활성 링크가 없으면(교체 전이거나 스타일링 팁) None 이다.
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

        link_ids = self._active_link_ids(key)
        recommended = [
            {**fit, "product_id": link_ids.get(fit["keyword"])}
            for fit in _merge(sections, "recommended")
        ]
        return {
            "body_type": body_type,
            "body_type_label": entry["label"],
            "description": entry["description"],
            "secondary": secondary,
            "recommended": recommended,
            "avoid": _merge(sections, "avoid"),
            "disclosure": self.disclosure,
        }


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
