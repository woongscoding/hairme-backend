"""
템플릿 기반 헤어스타일 추천 이유 생성기

얼굴형, 피부톤, 헤어스타일 조합에 대한 자연스러운 추천 이유를 생성합니다.

Author: HairMe Team
Date: 2025-11-09
Version: 1.0.0
"""

import random
import logging
from typing import Optional, Dict, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ReasonGenerator:
    """템플릿 기반 추천 이유 생성기"""

    # ========== 얼굴형별 템플릿 ==========
    FACE_TEMPLATES = {
        "계란형": [
            "{style}로 이상적인 얼굴 비율을 더욱 돋보이게 합니다",
            "{style}이(가) 균형잡힌 얼굴형과 완벽하게 조화를 이룹니다",
            "{style}로 우아한 얼굴 라인을 살려줍니다",
            "{style}이(가) 어떤 스타일이든 잘 어울리는 얼굴형에 적합합니다",
        ],
        "둥근형": [
            "{style}로 갸름한 얼굴선을 연출합니다",
            "{style}이(가) 동그란 윤곽을 시각적으로 보완해줍니다",
            "{style}로 얼굴 길이감을 더해 세련된 인상을 줍니다",
            "{style}이(가) 볼륨감을 조절해 슬림한 라인을 만들어줍니다",
        ],
        "각진형": [
            "{style}로 각진 턱선을 부드럽게 커버합니다",
            "{style}이(가) 강한 윤곽을 자연스럽게 중화시켜줍니다",
            "{style}로 부드러운 인상을 연출할 수 있습니다",
            "{style}이(가) 각진 페이스 라인에 여성스러움을 더합니다",
        ],
        "긴형": [
            "{style}로 얼굴 길이의 균형을 맞춰줍니다",
            "{style}이(가) 세로 비율을 시각적으로 조절해줍니다",
            "{style}로 긴 얼굴을 컴팩트하게 보이게 합니다",
            "{style}이(가) 얼굴 황금비율에 가깝게 만들어줍니다",
        ],
        "하트형": [
            "{style}로 좁은 턱선을 보완하고 균형을 맞춥니다",
            "{style}이(가) 넓은 이마와 조화를 이루는 스타일입니다",
            "{style}로 하관부에 볼륨을 더해 안정감을 줍니다",
            "{style}이(가) 사랑스러운 하트형 얼굴에 잘 어울립니다",
        ],
    }

    # ========== 피부톤별 보조 설명 ==========
    SKIN_TONE_COMPLEMENTS = {
        "봄웜": [
            "따뜻한 피부톤과 잘 어울립니다",
            "봄 웜톤 피부를 더욱 화사하게 만들어줍니다",
            "밝고 따뜻한 피부톤과 조화롭습니다",
        ],
        "가을웜": [
            "깊이 있는 웜톤 피부와 잘 맞습니다",
            "가을 웜톤 피부를 더욱 고급스럽게 보이게 합니다",
            "차분한 웜톤 피부와 조화를 이룹니다",
        ],
        "여름쿨": [
            "시원한 피부톤에 청량한 느낌을 더합니다",
            "여름 쿨톤 피부를 더욱 청아하게 만들어줍니다",
            "부드러운 쿨톤 피부와 잘 어울립니다",
        ],
        "겨울쿨": [
            "선명한 쿨톤 피부와 완벽하게 어울립니다",
            "겨울 쿨톤 피부를 더욱 도드라지게 만들어줍니다",
            "차가운 피부톤과 멋진 조화를 이룹니다",
        ],
    }

    # ========== 스타일 특성 (추후 확장 가능) ==========
    STYLE_CHARACTERISTICS: Dict[str, List[str]] = {}

    def __init__(self, characteristics_path: Optional[str] = None):
        """
        초기화

        Args:
            characteristics_path: 스타일 특성 JSON 파일 경로 (선택)
        """
        self.characteristics_path = characteristics_path

        # 스타일 특성 파일이 있으면 로드
        if characteristics_path:
            self._load_characteristics(characteristics_path)

    def _load_characteristics(self, path: str):
        """스타일 특성 데이터 로드"""
        try:
            file_path = Path(path)
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.STYLE_CHARACTERISTICS = data
                logger.info(
                    f"✅ 스타일 특성 로드 완료: {len(self.STYLE_CHARACTERISTICS)}개"
                )
            else:
                logger.warning(f"⚠️ 스타일 특성 파일 없음: {path}")
        except Exception as e:
            logger.error(f"❌ 스타일 특성 로드 실패: {str(e)}")

    def generate(
        self,
        face_shape: str,
        skin_tone: str,
        hairstyle: str,
        include_skin_tone: bool = True,
    ) -> str:
        """
        추천 이유 생성

        Args:
            face_shape: 얼굴형 (계란형, 둥근형, 각진형, 긴형, 하트형)
            skin_tone: 피부톤 (봄웜, 가을웜, 여름쿨, 겨울쿨)
            hairstyle: 헤어스타일명
            include_skin_tone: 피부톤 설명 포함 여부

        Returns:
            추천 이유 문자열 (30자 내외)
        """
        # 1. 얼굴형 템플릿 선택
        face_templates = self.FACE_TEMPLATES.get(
            face_shape, self.FACE_TEMPLATES["계란형"]  # 기본값
        )
        face_reason = random.choice(face_templates).format(style=hairstyle)

        # 2. 스타일 특성 추가 (있는 경우)
        characteristics = self.STYLE_CHARACTERISTICS.get(hairstyle, [])
        if characteristics:
            # 특성 1-2개만 사용
            char_text = ", ".join(characteristics[:2])
            face_reason += f" ({char_text})"

        # 3. 피부톤 보조 설명 추가 (옵션)
        if include_skin_tone:
            skin_complements = self.SKIN_TONE_COMPLEMENTS.get(skin_tone, [])
            if skin_complements:
                skin_text = random.choice(skin_complements)
                # 길이 체크 (총 60자 이내)
                combined = f"{face_reason}. {skin_text}"
                if len(combined) <= 60:
                    return combined

        return face_reason

    def generate_simple(self, face_shape: str, skin_tone: str, hairstyle: str) -> str:
        """
        간단한 버전 (피부톤 설명 제외, 30자 이내)

        Args:
            face_shape: 얼굴형
            skin_tone: 피부톤 (사용 안함)
            hairstyle: 헤어스타일명

        Returns:
            추천 이유 (30자 이내)
        """
        face_templates = self.FACE_TEMPLATES.get(
            face_shape, self.FACE_TEMPLATES["계란형"]
        )
        return random.choice(face_templates).format(style=hairstyle)

    def generate_with_score(
        self, face_shape: str, skin_tone: str, hairstyle: str, ml_score: float
    ) -> str:
        """
        ML 점수를 포함한 이유 생성

        Args:
            face_shape: 얼굴형
            skin_tone: 피부톤
            hairstyle: 헤어스타일명
            ml_score: ML 예측 점수 (0-100)

        Returns:
            추천 이유 + 신뢰도
        """
        base_reason = self.generate_simple(face_shape, skin_tone, hairstyle)

        # 점수별 신뢰도 표현 (더 자연스럽게 수정)
        if ml_score >= 90:
            confidence = "★★★ 강력 추천"
        elif ml_score >= 85:
            confidence = "★★★ 매우 잘 어울림"
        elif ml_score >= 80:
            confidence = "★★☆ 추천"
        elif ml_score >= 75:
            confidence = "★★☆ 잘 어울림"
        else:
            confidence = "★☆☆ 추천"

        # "AI"라는 단어를 제거하고 더 자연스럽게 표현
        return f"{base_reason} {confidence}"

    def add_characteristic(self, hairstyle: str, characteristics: List[str]):
        """
        특정 스타일의 특성 추가

        Args:
            hairstyle: 헤어스타일명
            characteristics: 특성 리스트 (예: ["입체감", "볼륨감"])
        """
        self.STYLE_CHARACTERISTICS[hairstyle] = characteristics

    def save_characteristics(self, path: str):
        """
        스타일 특성을 JSON 파일로 저장

        Args:
            path: 저장할 파일 경로
        """
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.STYLE_CHARACTERISTICS, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 스타일 특성 저장 완료: {path}")
        except Exception as e:
            logger.error(f"❌ 스타일 특성 저장 실패: {str(e)}")


# ========== 싱글톤 인스턴스 ==========
_reason_generator_instance = None


def get_reason_generator(
    characteristics_path: Optional[str] = "data_source/style_characteristics.json",
) -> ReasonGenerator:
    """
    ReasonGenerator 싱글톤 인스턴스 가져오기

    Args:
        characteristics_path: 스타일 특성 JSON 파일 경로

    Returns:
        ReasonGenerator 인스턴스
    """
    global _reason_generator_instance

    if _reason_generator_instance is None:
        logger.info("🔧 추천 이유 생성기 초기화 중...")
        _reason_generator_instance = ReasonGenerator(characteristics_path)
        logger.info("✅ 추천 이유 생성기 준비 완료")

    return _reason_generator_instance


# ========== 테스트용 ==========
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)

    # 생성기 초기화
    generator = ReasonGenerator()

    # 테스트
    test_cases = [
        ("계란형", "봄웜", "레이어드 컷"),
        ("둥근형", "가을웜", "단발 보브"),
        ("각진형", "여름쿨", "시스루뱅"),
        ("긴형", "겨울쿨", "허쉬컷"),
        ("하트형", "봄웜", "웨이브 펌"),
    ]

    print("\n=== 추천 이유 생성 테스트 ===\n")

    for face, skin, style in test_cases:
        reason = generator.generate(face, skin, style)
        print(f"[{face}] + [{skin}] + [{style}]")
        print(f"  → {reason}")
        print()

    # ML 점수 포함 테스트
    print("\n=== ML 점수 포함 테스트 ===\n")
    for score in [95, 85, 75, 65]:
        reason = generator.generate_with_score("계란형", "봄웜", "레이어드 컷", score)
        print(f"점수 {score}: {reason}")
