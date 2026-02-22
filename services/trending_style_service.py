"""
트렌드 헤어스타일 선택 서비스

ML 추천에 포함되지 않은 트렌드 스타일을 랜덤으로 선택하여
추천 다양성을 높입니다.

Author: HairMe ML Team
Date: 2025-02
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Set, Any

from utils.style_preprocessor import normalize_style_name

logger = logging.getLogger(__name__)

# 트렌드 데이터 경로
_TRENDING_DATA_PATH = Path(__file__).parent.parent / "data_source" / "trending_hairstyles.json"

# 트렌드 추천 이유 템플릿
_TRENDING_REASONS = [
    "최근 인기 트렌드 스타일",
    "올해 주목받는 스타일",
    "트렌디한 스타일 추천",
]


class TrendingStyleService:
    """트렌드 헤어스타일 선택 서비스"""

    def __init__(self):
        self.styles: Dict[str, List[str]] = {"male": [], "female": [], "unisex": []}
        self._load_data()

    def _load_data(self):
        """JSON 파일에서 트렌드 스타일 로드"""
        try:
            with open(_TRENDING_DATA_PATH, "r", encoding="utf-8") as f:
                self.styles = json.load(f)
            total = sum(len(v) for v in self.styles.values())
            logger.info(f"트렌드 스타일 로드 완료: {total}개")
        except Exception as e:
            logger.error(f"트렌드 스타일 로드 실패: {e}")

    def _get_candidates(self, gender: str) -> List[str]:
        """성별에 맞는 후보 스타일 반환 (unisex 포함)"""
        unisex = list(self.styles.get("unisex", []))

        if gender == "male":
            return list(self.styles.get("male", [])) + unisex
        elif gender == "female":
            return list(self.styles.get("female", [])) + unisex
        else:
            # neutral / unknown: 전체
            return (
                list(self.styles.get("male", []))
                + list(self.styles.get("female", []))
                + unisex
            )

    def pick_trending(
        self,
        gender: str,
        exclude_styles: Set[str],
        count: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        ML 추천과 중복되지 않는 트렌드 스타일 선택

        Args:
            gender: 성별 ("male", "female", "neutral")
            exclude_styles: 제외할 스타일 이름 집합 (정규화된 이름)
            count: 선택 개수 (기본 2)

        Returns:
            트렌드 추천 리스트
            [{"style_name": ..., "source": "trending", "reason": ..., "score": None}, ...]
        """
        candidates = self._get_candidates(gender)

        # ML 추천과 중복 제거 (정규화 기준)
        filtered = [
            s for s in candidates
            if normalize_style_name(s) not in exclude_styles
        ]

        if not filtered:
            logger.warning("트렌드 후보가 없음 (모두 ML 추천과 중복)")
            return []

        picked = random.sample(filtered, min(count, len(filtered)))

        results = []
        for style_name in picked:
            results.append({
                "hairstyle_id": None,
                "style_name": style_name,
                "source": "trending",
                "reason": random.choice(_TRENDING_REASONS),
                "score": None,
            })

        logger.info(f"트렌드 스타일 선택: {[r['style_name'] for r in results]}")
        return results


# ========== 싱글톤 인스턴스 ==========
_trending_service_instance = None


def get_trending_style_service() -> TrendingStyleService:
    """트렌드 스타일 서비스 싱글톤 인스턴스"""
    global _trending_service_instance

    if _trending_service_instance is None:
        _trending_service_instance = TrendingStyleService()

    return _trending_service_instance
