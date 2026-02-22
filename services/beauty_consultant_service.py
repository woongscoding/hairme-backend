"""BeautyMe Integrated Consultant Service

종합 뷰티 컨설팅 서비스
- 얼굴 분석 (얼굴형 + 퍼스널컬러)
- 헤어스타일 추천
- 염색 추천
- 스타일링 조언
- AI 상담
"""

import time
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from core.logging import logger

# Type hints only - no runtime import (Lambda cold start optimization)
if TYPE_CHECKING:
    from models.mediapipe_analyzer import MediaPipeFaceAnalyzer, MediaPipeFaceFeatures


@dataclass
class BeautyProfile:
    """사용자 뷰티 프로필"""
    # 기본 분석
    face_shape: str
    personal_color: str
    gender: str
    confidence: float

    # 상세 분석값
    ita_value: float = 0.0
    hue_value: float = 0.0
    face_ratio: float = 0.0

    # 추천
    hairstyles: List[Dict[str, Any]] = field(default_factory=list)
    hair_colors: List[Dict[str, Any]] = field(default_factory=list)
    color_palette: List[Dict[str, str]] = field(default_factory=list)

    # 스타일링 조언
    styling_tips: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": {
                "face_shape": self.face_shape,
                "personal_color": self.personal_color,
                "gender": self.gender,
                "confidence": self.confidence
            },
            "analysis": {
                "ita_value": self.ita_value,
                "hue_value": self.hue_value,
                "face_ratio": self.face_ratio
            },
            "recommendations": {
                "hairstyles": self.hairstyles,
                "hair_colors": self.hair_colors,
                "color_palette": self.color_palette
            },
            "styling": self.styling_tips
        }


class BeautyConsultantService:
    """
    BeautyMe 통합 뷰티 컨설턴트 서비스

    Features:
    - 원스톱 얼굴 분석 (얼굴형 + 퍼스널컬러)
    - 통합 추천 (헤어스타일 + 염색 + 스타일링)
    - AI 기반 맞춤 상담
    """

    def __init__(self):
        self._analyzer = None  # MediaPipeFaceAnalyzer (lazy loaded)
        self._personal_color_service = None
        self._hair_color_service = None
        self._ml_recommender = None
        self._chatbot_service = None

    @property
    def analyzer(self):
        """Lazy load MediaPipe analyzer"""
        if self._analyzer is None:
            # Lazy import to reduce Lambda cold start time
            from models.mediapipe_analyzer import MediaPipeFaceAnalyzer
            self._analyzer = MediaPipeFaceAnalyzer()
            logger.info("✅ BeautyConsultant: MediaPipe analyzer loaded")
        return self._analyzer

    @property
    def personal_color_service(self):
        """Lazy load personal color service"""
        if self._personal_color_service is None:
            from services.personal_color_service import get_personal_color_service
            self._personal_color_service = get_personal_color_service()
        return self._personal_color_service

    @property
    def hair_color_service(self):
        """Lazy load hair color service"""
        if self._hair_color_service is None:
            from services.hair_color_service import get_hair_color_service
            self._hair_color_service = get_hair_color_service()
        return self._hair_color_service

    @property
    def ml_recommender(self):
        """Lazy load ML recommender"""
        if self._ml_recommender is None:
            from core.dependencies import get_hybrid_service
            self._ml_recommender = get_hybrid_service()
        return self._ml_recommender

    @property
    def chatbot_service(self):
        """Lazy load chatbot service"""
        if self._chatbot_service is None:
            from services.chatbot_service import get_chatbot_service
            self._chatbot_service = get_chatbot_service()
        return self._chatbot_service

    def analyze_full(
        self,
        image_data: bytes,
        gender: str = "neutral"
    ) -> Optional[BeautyProfile]:
        """
        종합 뷰티 분석

        이미지 한 장으로 얼굴형, 퍼스널컬러를 분석하고
        헤어스타일, 염색, 스타일링 추천을 모두 제공합니다.

        Args:
            image_data: 얼굴 이미지 bytes
            gender: 성별 (male/female/neutral)

        Returns:
            BeautyProfile with all recommendations
        """
        start_time = time.time()

        try:
            # 1. MediaPipe 얼굴 분석
            features = self.analyzer.analyze(image_data)

            if not features:
                logger.warning("BeautyConsultant: Face analysis failed")
                return None

            face_shape = features.face_shape
            personal_color = features.skin_tone
            detected_gender = features.gender if hasattr(features, 'gender') else gender
            final_gender = gender if gender != "neutral" else detected_gender

            logger.info(
                f"✅ 얼굴 분석 완료: {face_shape}, {personal_color}, {final_gender}"
            )

            # 2. 헤어스타일 추천 (ML 기반)
            hairstyles = []
            try:
                ml_result = self.ml_recommender.recommend(
                    image_data=image_data,
                    face_shape=face_shape,
                    skin_tone=personal_color,
                    face_features=features.face_features,
                    skin_features=features.skin_features,
                    gender=final_gender
                )
                hairstyles = ml_result.get("recommendations", [])[:3]
                logger.info(f"✅ 헤어스타일 추천: {len(hairstyles)}개")
            except Exception as e:
                logger.warning(f"헤어스타일 추천 실패: {e}")

            # 3. 염색 추천
            hair_colors = []
            try:
                color_result = self.hair_color_service.get_recommendations(
                    personal_color, include_trends=True
                )
                # 추천 + 트렌드 합쳐서 상위 5개
                for rec in color_result.recommended[:3]:
                    hair_colors.append({
                        "name": rec.name,
                        "hex": rec.hex,
                        "description": rec.description,
                        "is_trend": False
                    })
                for trend in color_result.trends[:2]:
                    hair_colors.append({
                        "name": trend.name,
                        "hex": trend.hex,
                        "description": trend.description,
                        "is_trend": True
                    })
                logger.info(f"✅ 염색 추천: {len(hair_colors)}개")
            except Exception as e:
                logger.warning(f"염색 추천 실패: {e}")

            # 4. 컬러 팔레트
            color_palette = []
            try:
                palette = self.personal_color_service.get_color_palette(personal_color)
                color_palette = palette[:6]  # 상위 6개
                logger.info(f"✅ 컬러 팔레트: {len(color_palette)}개")
            except Exception as e:
                logger.warning(f"컬러 팔레트 조회 실패: {e}")

            # 5. 스타일링 조언
            styling_tips = {}
            try:
                tips = self.personal_color_service.get_styling_tips(personal_color)
                styling_tips = {
                    "makeup": tips.get("makeup_tips", []),
                    "fashion": tips.get("fashion_tips", []),
                    "description": tips.get("description", "")
                }
                logger.info("✅ 스타일링 조언 로드 완료")
            except Exception as e:
                logger.warning(f"스타일링 조언 조회 실패: {e}")

            # 6. BeautyProfile 생성
            profile = BeautyProfile(
                face_shape=face_shape,
                personal_color=personal_color,
                gender=final_gender,
                confidence=features.confidence,
                ita_value=features.ITA_value,
                hue_value=features.hue_value,
                face_ratio=features.face_ratio,
                hairstyles=hairstyles,
                hair_colors=hair_colors,
                color_palette=color_palette,
                styling_tips=styling_tips
            )

            elapsed = round(time.time() - start_time, 2)
            logger.info(f"✅ 종합 뷰티 분석 완료 ({elapsed}초)")

            return profile

        except Exception as e:
            logger.error(f"❌ 종합 뷰티 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_consultation(
        self,
        query: str,
        profile: Optional[BeautyProfile] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        AI 뷰티 상담

        사용자 질문에 대해 프로필 정보를 활용한 맞춤 상담을 제공합니다.

        Args:
            query: 사용자 질문
            profile: 분석된 BeautyProfile (있으면 맞춤 상담)
            session_id: 대화 세션 ID

        Returns:
            상담 결과
        """
        try:
            # 프로필이 있으면 user_profile로 변환
            user_profile = None
            if profile:
                user_profile = {
                    "face_shape": profile.face_shape,
                    "personal_color": profile.personal_color,
                    "gender": profile.gender
                }

            # 챗봇 서비스 호출
            response = self.chatbot_service.chat(
                query=query,
                user_profile=user_profile,
                session_id=session_id
            )

            return {
                "success": response.success,
                "message": response.message,
                "intent": response.intent,
                "suggestions": response.suggestions,
                "sources": response.sources
            }

        except Exception as e:
            logger.error(f"❌ AI 상담 실패: {e}")
            return {
                "success": False,
                "message": "상담 중 오류가 발생했습니다.",
                "intent": "error",
                "suggestions": [],
                "sources": []
            }

    def generate_report(self, profile: BeautyProfile) -> str:
        """
        뷰티 리포트 생성

        분석 결과를 읽기 좋은 리포트 형식으로 생성합니다.

        Args:
            profile: BeautyProfile

        Returns:
            마크다운 형식의 리포트
        """
        report = f"""# 🌸 BeautyMe 분석 리포트

## 기본 분석 결과

| 항목 | 결과 |
|------|------|
| 얼굴형 | {profile.face_shape} |
| 퍼스널컬러 | {profile.personal_color} |
| 분석 신뢰도 | {profile.confidence:.0%} |

---

## 💇 헤어스타일 추천

"""
        for i, style in enumerate(profile.hairstyles[:3], 1):
            report += f"{i}. **{style.get('style_name', 'Unknown')}**\n"
            if style.get('reason'):
                report += f"   - {style.get('reason')}\n"

        report += """
---

## 🎨 염색 추천

"""
        for color in profile.hair_colors[:5]:
            trend_badge = " 🔥트렌드" if color.get('is_trend') else ""
            report += f"- **{color.get('name')}** ({color.get('hex')}){trend_badge}\n"
            if color.get('description'):
                report += f"  - {color.get('description')}\n"

        report += """
---

## 👗 스타일링 조언

### 메이크업
"""
        for tip in profile.styling_tips.get('makeup', [])[:3]:
            report += f"- {tip}\n"

        report += """
### 패션
"""
        for tip in profile.styling_tips.get('fashion', [])[:3]:
            report += f"- {tip}\n"

        report += """
---

*분석 by BeautyMe AI*
"""
        return report


# Singleton
_beauty_consultant: Optional[BeautyConsultantService] = None


def get_beauty_consultant_service() -> BeautyConsultantService:
    """BeautyConsultantService 싱글톤 인스턴스"""
    global _beauty_consultant
    if _beauty_consultant is None:
        _beauty_consultant = BeautyConsultantService()
        logger.info("✅ BeautyConsultantService initialized")
    return _beauty_consultant
