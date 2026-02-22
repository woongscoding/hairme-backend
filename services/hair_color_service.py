"""Hair Color Recommendation and Synthesis Service

Phase 3: 염색 추천 + 합성 서비스
- 퍼스널컬러 기반 염색 추천
- 트렌드 염색 컬러 제공
- 가상 염색 시뮬레이션
"""

import base64
import io
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from PIL import Image

from core.logging import logger
from config.settings import settings


@dataclass
class HairColorRecommendation:
    """염색 추천 결과"""
    name: str
    hex: str
    level: str  # 밝기 레벨 (1-10)
    description: str
    suitable_for: List[str] = field(default_factory=list)
    is_trend: bool = False


@dataclass
class HairColorResult:
    """염색 추천 전체 결과"""
    personal_color: str
    recommended: List[HairColorRecommendation]
    avoid: List[Dict[str, str]]
    trends: List[HairColorRecommendation]


class HairColorService:
    """염색 추천 및 합성 서비스"""

    DATA_DIR = Path("data/lookup")

    # 2024-2025 트렌드 염색 컬러
    TREND_COLORS = {
        "2024_winter": [
            {
                "name": "글레이즈 브라운",
                "hex": "#8B6914",
                "level": "6-7",
                "description": "광택감 있는 글로시 브라운, 2024년 겨울 인기",
                "suitable_for": ["모든 퍼스널컬러"],
                "trend_type": "universal"
            },
            {
                "name": "체리 브라운",
                "hex": "#5C1E1E",
                "level": "4-5",
                "description": "체리빛 레드 브라운, 여성스러운 분위기",
                "suitable_for": ["가을웜", "겨울쿨"],
                "trend_type": "feminine"
            },
            {
                "name": "모카 베이지",
                "hex": "#9C8B76",
                "level": "7-8",
                "description": "부드러운 모카 베이지, 자연스러운 고급감",
                "suitable_for": ["봄웜", "가을웜"],
                "trend_type": "natural"
            }
        ],
        "2025_spring": [
            {
                "name": "테라코타 브라운",
                "hex": "#CC5A3C",
                "level": "5-6",
                "description": "테라코타 느낌의 웜브라운, 2025 S/S 트렌드",
                "suitable_for": ["봄웜", "가을웜"],
                "trend_type": "seasonal"
            },
            {
                "name": "아이시 라벤더",
                "hex": "#C9B8DB",
                "level": "9-10",
                "description": "시원한 라벤더 컬러, 개성있는 스타일",
                "suitable_for": ["여름쿨", "겨울쿨"],
                "trend_type": "bold"
            },
            {
                "name": "페일 핑크",
                "hex": "#EACACB",
                "level": "9-10",
                "description": "연한 핑크빛 블론드, 로맨틱한 분위기",
                "suitable_for": ["여름쿨", "봄웜"],
                "trend_type": "romantic"
            },
            {
                "name": "샴페인 골드",
                "hex": "#D4AF37",
                "level": "8-9",
                "description": "샴페인처럼 우아한 골드, 화사한 이미지",
                "suitable_for": ["봄웜"],
                "trend_type": "elegant"
            },
            {
                "name": "딥 버건디",
                "hex": "#4A0E0E",
                "level": "2-3",
                "description": "깊은 와인색, 시크하고 고급스러운 느낌",
                "suitable_for": ["가을웜", "겨울쿨"],
                "trend_type": "chic"
            }
        ]
    }

    def __init__(self):
        self._hair_color_data: Optional[Dict] = None
        self._gemini_client = None

    @property
    def hair_color_data(self) -> Dict:
        """Load hair color data"""
        if self._hair_color_data is None:
            try:
                with open(self.DATA_DIR / "pc_hair_color.json", 'r', encoding='utf-8') as f:
                    self._hair_color_data = json.load(f)
                logger.info("✅ Hair color data loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load hair color data: {e}")
                self._hair_color_data = {}
        return self._hair_color_data

    @property
    def gemini_client(self):
        """Lazy load Gemini client"""
        if self._gemini_client is None:
            from google import genai
            self._gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
            logger.info("✅ HairColorService: Gemini client loaded")
        return self._gemini_client

    def get_recommendations(
        self,
        personal_color: str,
        include_trends: bool = True
    ) -> HairColorResult:
        """
        퍼스널컬러 기반 염색 추천

        Args:
            personal_color: 퍼스널컬러 (봄웜/여름쿨/가을웜/겨울쿨)
            include_trends: 트렌드 컬러 포함 여부

        Returns:
            HairColorResult
        """
        pc_data = self.hair_color_data.get(personal_color, {})

        # 추천 컬러
        recommended = []
        for color in pc_data.get("recommended", []):
            recommended.append(HairColorRecommendation(
                name=color.get("name", ""),
                hex=color.get("hex", "#000000"),
                level=color.get("level", "5"),
                description=color.get("description", ""),
                suitable_for=color.get("suitable_for", [])
            ))

        # 피해야 할 컬러
        avoid = pc_data.get("avoid", [])

        # 트렌드 컬러 (퍼스널컬러에 맞는 것만)
        trends = []
        if include_trends:
            trends = self._get_matching_trends(personal_color)

        return HairColorResult(
            personal_color=personal_color,
            recommended=recommended,
            avoid=avoid,
            trends=trends
        )

    def _get_matching_trends(self, personal_color: str) -> List[HairColorRecommendation]:
        """퍼스널컬러에 맞는 트렌드 컬러 필터링"""
        matching = []

        for season, colors in self.TREND_COLORS.items():
            for color in colors:
                suitable = color.get("suitable_for", [])
                # "모든 퍼스널컬러"이거나 해당 퍼스널컬러가 포함된 경우
                if "모든 퍼스널컬러" in suitable or personal_color in suitable:
                    matching.append(HairColorRecommendation(
                        name=color["name"],
                        hex=color["hex"],
                        level=color["level"],
                        description=color["description"],
                        suitable_for=suitable,
                        is_trend=True
                    ))

        return matching

    def get_all_trends(self) -> Dict[str, List[Dict]]:
        """모든 트렌드 컬러 조회"""
        return self.TREND_COLORS

    def synthesize_hair_color(
        self,
        image_data: bytes,
        color_name: str,
        color_hex: str,
        additional_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        가상 염색 시뮬레이션

        Args:
            image_data: 사용자 사진 bytes
            color_name: 염색 컬러명 (예: "밀크브라운")
            color_hex: HEX 코드 (예: "#C4A484")
            additional_instructions: 추가 요청사항

        Returns:
            {
                "success": bool,
                "image_base64": str,
                "image_format": str,
                "message": str
            }
        """
        try:
            from google.genai import types

            # Open image
            original_image = Image.open(io.BytesIO(image_data))

            # Build prompt (단순화 - 속도 개선)
            prompt = f"""이 사진의 인물에게 '{color_name}' 염색을 적용해주세요.

요구사항:
1. 인물의 얼굴은 그대로 유지하고, 머리카락 색상만 '{color_name}' ({color_hex})로 자연스럽게 변경해주세요.
2. 머리카락의 질감이 자연스럽게 보이도록 해주세요.
3. 고품질의 사실적인 이미지를 생성해주세요."""

            if additional_instructions:
                prompt += f"\n4. 추가 요청: {additional_instructions}"

            logger.info(f"🎨 염색 시뮬레이션 시작: {color_name} ({color_hex})")

            # Call Gemini API with retry
            MAX_RETRIES = 3
            RETRY_DELAY = 1.0

            result_image = None
            result_text = None

            for attempt in range(MAX_RETRIES):
                try:
                    response = self.gemini_client.models.generate_content(
                        model="gemini-2.5-flash-image",
                        contents=[prompt, original_image],
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE", "TEXT"],
                        )
                    )

                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            result_image = part.inline_data
                        elif hasattr(part, 'text') and part.text:
                            result_text = part.text

                    if result_image is not None:
                        logger.info(f"✅ API 호출 성공 (시도 {attempt + 1}/{MAX_RETRIES})")
                        break
                    else:
                        logger.warning(f"⚠️ 이미지 미생성 (시도 {attempt + 1})")
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(RETRY_DELAY)

                except Exception as api_error:
                    logger.warning(f"⚠️ API 호출 실패 (시도 {attempt + 1}): {str(api_error)}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)

            if result_image is None:
                return {
                    "success": False,
                    "image_base64": None,
                    "image_format": None,
                    "message": "염색 시뮬레이션 생성에 실패했습니다. 다시 시도해주세요."
                }

            # Convert to base64
            image_bytes = result_image.data

            if isinstance(image_bytes, str):
                image_base64 = image_bytes
            elif isinstance(image_bytes, bytes):
                if image_bytes[:4] == b'\x89PNG' or image_bytes[:3] == b'\xff\xd8\xff':
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                else:
                    try:
                        decoded_str = image_bytes.decode('utf-8')
                        if decoded_str.startswith('iVBOR') or decoded_str.startswith('/9j/'):
                            image_base64 = decoded_str
                        else:
                            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    except UnicodeDecodeError:
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            else:
                image_base64 = base64.b64encode(bytes(image_bytes)).decode('utf-8')

            mime_type = result_image.mime_type
            image_format = mime_type.split('/')[-1] if mime_type else "png"

            logger.info(f"✅ 염색 시뮬레이션 완료: {color_name}")

            return {
                "success": True,
                "image_base64": image_base64,
                "image_format": image_format,
                "message": f"'{color_name}' 염색이 적용되었습니다.",
                "gemini_response": result_text
            }

        except Exception as e:
            logger.error(f"❌ 염색 시뮬레이션 실패: {str(e)}")
            import traceback
            traceback.print_exc()

            return {
                "success": False,
                "image_base64": None,
                "image_format": None,
                "message": f"염색 시뮬레이션 중 오류: {str(e)}"
            }

    def get_color_by_name(self, color_name: str) -> Optional[Dict[str, Any]]:
        """컬러명으로 컬러 정보 조회"""
        # 모든 퍼스널컬러 데이터에서 검색
        for pc_type, data in self.hair_color_data.items():
            for color in data.get("recommended", []):
                if color.get("name") == color_name:
                    return {**color, "personal_color": pc_type}

        # 트렌드 컬러에서 검색
        for season, colors in self.TREND_COLORS.items():
            for color in colors:
                if color.get("name") == color_name:
                    return {**color, "season": season, "is_trend": True}

        return None


# Singleton
_hair_color_service: Optional[HairColorService] = None


def get_hair_color_service() -> HairColorService:
    """HairColorService 싱글톤 인스턴스"""
    global _hair_color_service
    if _hair_color_service is None:
        _hair_color_service = HairColorService()
        logger.info("✅ HairColorService initialized")
    return _hair_color_service
