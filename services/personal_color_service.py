"""Personal Color Analysis Service

Phase 2: 퍼스널컬러 분석 서비스
- MediaPipe 피부톤 분석 활용
- 상세 컬러 팔레트 제공
- 스타일링 조언 제공
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from core.logging import logger

# Type hints only - no runtime import (Lambda cold start optimization)
if TYPE_CHECKING:
    from models.mediapipe_analyzer import MediaPipeFaceAnalyzer, MediaPipeFaceFeatures


@dataclass
class PersonalColorResult:
    """퍼스널컬러 분석 결과"""
    personal_color: str  # 봄웜, 여름쿨, 가을웜, 겨울쿨
    confidence: float
    season: str  # spring, summer, autumn, winter
    tone: str  # warm, cool

    # 분석 상세
    ita_value: float  # ITA (Individual Typology Angle)
    hue_value: float  # HSV Hue
    brightness: str  # bright, muted, clear
    undertone: str  # yellow, pink, golden, blue

    # 추천 정보
    characteristics: List[str] = field(default_factory=list)
    best_colors: List[Dict[str, str]] = field(default_factory=list)  # [{name, hex, description}]
    avoid_colors: List[str] = field(default_factory=list)
    hair_colors: List[Dict[str, str]] = field(default_factory=list)  # 추천 염색

    # 스타일링 조언
    makeup_tips: List[str] = field(default_factory=list)
    fashion_tips: List[str] = field(default_factory=list)
    styling_description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "personal_color": self.personal_color,
            "confidence": self.confidence,
            "season": self.season,
            "tone": self.tone,
            "analysis": {
                "ita_value": self.ita_value,
                "hue_value": self.hue_value,
                "brightness": self.brightness,
                "undertone": self.undertone
            },
            "characteristics": self.characteristics,
            "palette": {
                "best_colors": self.best_colors,
                "avoid_colors": self.avoid_colors,
                "hair_colors": self.hair_colors
            },
            "styling": {
                "makeup_tips": self.makeup_tips,
                "fashion_tips": self.fashion_tips,
                "description": self.styling_description
            }
        }


class PersonalColorService:
    """퍼스널컬러 분석 서비스"""

    DATA_DIR = Path("data/lookup")

    # 확장된 컬러 팔레트 (HEX 코드 포함)
    COLOR_PALETTES = {
        "봄웜": [
            {"name": "코랄 핑크", "hex": "#FF7F7F", "description": "화사한 산호색"},
            {"name": "피치", "hex": "#FFCBA4", "description": "복숭아 빛 따뜻한 색"},
            {"name": "아이보리", "hex": "#FFFFF0", "description": "부드러운 상아색"},
            {"name": "골드", "hex": "#FFD700", "description": "화려한 금색"},
            {"name": "살구색", "hex": "#FBCEB1", "description": "밝은 살구빛"},
            {"name": "연어색", "hex": "#FA8072", "description": "생기 있는 연어색"},
            {"name": "밝은 오렌지", "hex": "#FFA500", "description": "활기찬 오렌지"},
            {"name": "라이트 브라운", "hex": "#C4A484", "description": "따뜻한 연갈색"},
            {"name": "민트 그린", "hex": "#98FF98", "description": "상쾌한 민트색"},
            {"name": "카멜", "hex": "#C19A6B", "description": "고급스러운 카멜색"}
        ],
        "여름쿨": [
            {"name": "라벤더", "hex": "#E6E6FA", "description": "우아한 연보라"},
            {"name": "로즈 핑크", "hex": "#FF66B2", "description": "로맨틱한 장미색"},
            {"name": "스카이 블루", "hex": "#87CEEB", "description": "시원한 하늘색"},
            {"name": "민트", "hex": "#98FB98", "description": "청량한 민트색"},
            {"name": "소프트 화이트", "hex": "#F5F5F5", "description": "부드러운 흰색"},
            {"name": "그레이시 블루", "hex": "#6699CC", "description": "차분한 청회색"},
            {"name": "더스티 핑크", "hex": "#D8A9A9", "description": "뮤트한 핑크"},
            {"name": "페리윙클", "hex": "#CCCCFF", "description": "은은한 보랏빛"},
            {"name": "소프트 네이비", "hex": "#4169E1", "description": "부드러운 네이비"},
            {"name": "라이트 그레이", "hex": "#D3D3D3", "description": "밝은 회색"}
        ],
        "가을웜": [
            {"name": "카키", "hex": "#8B8B00", "description": "자연스러운 카키"},
            {"name": "머스타드", "hex": "#FFDB58", "description": "따뜻한 겨자색"},
            {"name": "테라코타", "hex": "#E2725B", "description": "흙빛 테라코타"},
            {"name": "버건디", "hex": "#800020", "description": "깊은 와인색"},
            {"name": "올리브", "hex": "#808000", "description": "은은한 올리브"},
            {"name": "캐멀", "hex": "#C19A6B", "description": "클래식한 낙타색"},
            {"name": "초콜릿", "hex": "#7B3F00", "description": "진한 초콜릿색"},
            {"name": "브릭", "hex": "#CB4154", "description": "벽돌색 레드"},
            {"name": "오렌지 브라운", "hex": "#CD853F", "description": "따뜻한 갈색"},
            {"name": "포레스트 그린", "hex": "#228B22", "description": "깊은 숲 녹색"}
        ],
        "겨울쿨": [
            {"name": "퓨어 화이트", "hex": "#FFFFFF", "description": "순수한 흰색"},
            {"name": "블랙", "hex": "#000000", "description": "세련된 검정"},
            {"name": "네이비", "hex": "#000080", "description": "클래식 네이비"},
            {"name": "트루 레드", "hex": "#FF0000", "description": "선명한 빨강"},
            {"name": "핫 핑크", "hex": "#FF69B4", "description": "강렬한 핑크"},
            {"name": "로얄 블루", "hex": "#4169E1", "description": "왕실의 파랑"},
            {"name": "에메랄드", "hex": "#50C878", "description": "선명한 에메랄드"},
            {"name": "버건디", "hex": "#722F37", "description": "깊은 버건디"},
            {"name": "실버", "hex": "#C0C0C0", "description": "차가운 은색"},
            {"name": "아이시 핑크", "hex": "#F8B9D4", "description": "차가운 핑크"}
        ]
    }

    # 패션 스타일링 조언
    FASHION_TIPS = {
        "봄웜": [
            "밝고 화사한 컬러의 원피스로 생기 있는 룩 연출",
            "골드 액세서리로 포인트",
            "베이지, 아이보리 계열 기본 아이템 추천",
            "파스텔 컬러 니트로 부드러운 이미지",
            "코랄 또는 피치 컬러 블라우스로 얼굴 화사하게"
        ],
        "여름쿨": [
            "시원한 파스텔톤 의상으로 청초한 이미지",
            "실버 액세서리가 얼굴을 밝게",
            "라벤더, 로즈핑크 컬러로 우아한 분위기",
            "그레이 계열 기본 아이템 추천",
            "소프트한 블루 계열로 청량감 연출"
        ],
        "가을웜": [
            "차분한 어스톤 컬러로 고급스러운 룩",
            "골드 또는 브론즈 액세서리 추천",
            "카키, 머스타드 컬러로 세련된 스타일",
            "브라운 계열 코트로 클래식한 분위기",
            "버건디 아이템으로 포인트"
        ],
        "겨울쿨": [
            "선명한 컬러 대비로 시크한 룩",
            "실버 또는 화이트골드 액세서리",
            "블랙 & 화이트 모노톤 스타일",
            "레드 또는 핫핑크로 강렬한 포인트",
            "네이비 수트로 세련된 비즈니스 룩"
        ]
    }

    def __init__(self):
        self._analyzer = None  # MediaPipeFaceAnalyzer (lazy loaded)
        self._personal_color_data: Optional[Dict] = None
        self._hair_color_data: Optional[Dict] = None

    @property
    def analyzer(self):
        """Lazy load MediaPipe analyzer"""
        if self._analyzer is None:
            # Lazy import to reduce Lambda cold start time
            from models.mediapipe_analyzer import MediaPipeFaceAnalyzer
            self._analyzer = MediaPipeFaceAnalyzer()
            logger.info("✅ PersonalColorService: MediaPipe analyzer loaded")
        return self._analyzer

    @property
    def personal_color_data(self) -> Dict:
        """Load personal color lookup data"""
        if self._personal_color_data is None:
            try:
                with open(self.DATA_DIR / "personal_color.json", 'r', encoding='utf-8') as f:
                    self._personal_color_data = json.load(f)
                logger.info("✅ Personal color data loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load personal color data: {e}")
                self._personal_color_data = {}
        return self._personal_color_data

    @property
    def hair_color_data(self) -> Dict:
        """Load hair color recommendation data"""
        if self._hair_color_data is None:
            try:
                with open(self.DATA_DIR / "pc_hair_color.json", 'r', encoding='utf-8') as f:
                    self._hair_color_data = json.load(f)
                logger.info("✅ Hair color data loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load hair color data: {e}")
                self._hair_color_data = {}
        return self._hair_color_data

    def analyze(self, image_data: bytes) -> Optional[PersonalColorResult]:
        """
        이미지에서 퍼스널컬러 분석

        Args:
            image_data: 이미지 바이트

        Returns:
            PersonalColorResult or None
        """
        try:
            # MediaPipe 분석
            features = self.analyzer.analyze(image_data)

            if not features:
                logger.warning("PersonalColorService: Face analysis failed")
                return None

            # 퍼스널컬러 정보 조회
            pc_info = self.personal_color_data.get(features.skin_tone, {})
            hair_info = self.hair_color_data.get(features.skin_tone, {})

            # 밝기 및 언더톤 결정
            brightness = self._determine_brightness(features.ITA_value)
            undertone = pc_info.get("skin_features", {}).get("undertone", "neutral")

            # 추천 염색 컬러
            hair_colors = []
            for hc in hair_info.get("recommended", [])[:5]:
                hair_colors.append({
                    "name": hc.get("name", ""),
                    "hex": hc.get("hex", "#000000"),
                    "description": hc.get("description", "")
                })

            # 결과 생성
            result = PersonalColorResult(
                personal_color=features.skin_tone,
                confidence=features.confidence,
                season=pc_info.get("season", "unknown"),
                tone=pc_info.get("tone", "unknown"),
                ita_value=features.ITA_value,
                hue_value=features.hue_value,
                brightness=brightness,
                undertone=undertone,
                characteristics=pc_info.get("characteristics", []),
                best_colors=self.COLOR_PALETTES.get(features.skin_tone, []),
                avoid_colors=pc_info.get("avoid_colors", []),
                hair_colors=hair_colors,
                makeup_tips=pc_info.get("makeup_tips", []),
                fashion_tips=self.FASHION_TIPS.get(features.skin_tone, []),
                styling_description=pc_info.get("description", "")
            )

            logger.info(
                f"✅ Personal color analysis complete: {features.skin_tone} "
                f"(ITA: {features.ITA_value:.1f}, Hue: {features.hue_value:.1f})"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Personal color analysis failed: {e}")
            return None

    def _determine_brightness(self, ita_value: float) -> str:
        """ITA 값으로 피부 밝기 결정"""
        if ita_value > 42:
            return "bright"  # 밝은 피부
        elif ita_value > 28:
            return "medium"  # 중간 피부
        else:
            return "muted"  # 어두운/차분한 피부

    def get_color_palette(self, personal_color: str) -> List[Dict[str, str]]:
        """특정 퍼스널컬러의 컬러 팔레트 반환"""
        return self.COLOR_PALETTES.get(personal_color, [])

    def get_styling_tips(self, personal_color: str) -> Dict[str, Any]:
        """특정 퍼스널컬러의 스타일링 팁 반환"""
        pc_info = self.personal_color_data.get(personal_color, {})
        return {
            "makeup_tips": pc_info.get("makeup_tips", []),
            "fashion_tips": self.FASHION_TIPS.get(personal_color, []),
            "description": pc_info.get("description", "")
        }

    def get_hair_recommendations(self, personal_color: str) -> Dict[str, Any]:
        """특정 퍼스널컬러의 염색 추천"""
        hair_info = self.hair_color_data.get(personal_color, {})
        return {
            "recommended": hair_info.get("recommended", []),
            "avoid": hair_info.get("avoid", [])
        }


# Singleton
_personal_color_service: Optional[PersonalColorService] = None


def get_personal_color_service() -> PersonalColorService:
    """PersonalColorService 싱글톤 인스턴스"""
    global _personal_color_service
    if _personal_color_service is None:
        _personal_color_service = PersonalColorService()
        logger.info("✅ PersonalColorService initialized")
    return _personal_color_service
