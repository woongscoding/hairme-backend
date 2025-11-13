"""
하이브리드 헤어스타일 추천 서비스

Gemini API + ML 모델을 결합하여 최적의 추천 제공

Author: HairMe ML Team
Date: 2025-11-08
Version: 1.0.0
"""

import logging
from typing import List, Dict, Optional
import google.generativeai as genai
from PIL import Image
import io
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.ml_recommender import get_ml_recommender
from services.reason_generator import get_reason_generator
from utils.style_preprocessor import normalize_style_name

logger = logging.getLogger(__name__)


class HybridRecommendationService:
    """Gemini + ML 하이브리드 추천 서비스"""

    def __init__(self, gemini_api_key: str):
        """
        초기화

        Args:
            gemini_api_key: Gemini API 키
        """
        # Gemini 설정
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
        logger.info("✅ Gemini 모델 초기화 완료")

        # ML 추천기는 싱글톤으로 로드
        try:
            self.ml_recommender = get_ml_recommender()
            self.ml_available = True
        except Exception as e:
            logger.error(f"❌ ML 추천기 로드 실패: {str(e)}")
            self.ml_recommender = None
            self.ml_available = False

        # 추천 이유 생성기 로드
        try:
            self.reason_generator = get_reason_generator()
            logger.info("✅ 추천 이유 생성기 초기화 완료")
        except Exception as e:
            logger.error(f"❌ 추천 이유 생성기 로드 실패: {str(e)}")
            self.reason_generator = None

    def _create_gemini_prompt(
        self,
        face_shape: str,
        skin_tone: str
    ) -> str:
        """
        Gemini API용 프롬프트 생성

        Args:
            face_shape: 얼굴형
            skin_tone: 피부톤

        Returns:
            프롬프트 문자열
        """
        prompt = f"""이 사람의 얼굴을 분석하고 헤어스타일을 추천해주세요.

**MediaPipe 분석 결과:**
- 얼굴형: {face_shape}
- 피부톤: {skin_tone}

다음 형식의 JSON으로만 응답:
{{
  "analysis": {{
    "face_shape": "{face_shape}",
    "personal_color": "{skin_tone}",
    "features": "이목구비 특징 (30자 이내)"
  }},
  "recommendations": [
    {{"style_name": "스타일명 (15자 이내)", "reason": "추천 이유 (30자 이내)"}},
    {{"style_name": "스타일명 (15자 이내)", "reason": "추천 이유 (30자 이내)"}},
    {{"style_name": "스타일명 (15자 이내)", "reason": "추천 이유 (30자 이내)"}},
    {{"style_name": "스타일명 (15자 이내)", "reason": "추천 이유 (30자 이내)"}}
  ]
}}

중요:
- 4개의 헤어스타일을 추천하세요
- 한국에서 실제로 사용하는 자연스러운 표현 사용
- 얼굴형과 피부톤에 가장 잘 어울리는 스타일 추천"""

        return prompt

    def _call_gemini(
        self,
        image_data: bytes,
        face_shape: str,
        skin_tone: str
    ) -> Dict:
        """
        Gemini API 호출

        Args:
            image_data: 이미지 바이트
            face_shape: 얼굴형
            skin_tone: 피부톤

        Returns:
            Gemini 분석 결과
        """
        try:
            # 이미지 로드
            image = Image.open(io.BytesIO(image_data))

            # 프롬프트 생성
            prompt = self._create_gemini_prompt(face_shape, skin_tone)

            # API 호출
            response = self.gemini_model.generate_content([prompt, image])

            # JSON 파싱
            import json
            raw_text = response.text.strip()

            # 마크다운 코드 블록 제거
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            result = json.loads(raw_text.strip())

            logger.info(f"✅ Gemini 응답: {len(result.get('recommendations', []))}개 추천")

            return result

        except Exception as e:
            logger.error(f"❌ Gemini API 오류: {str(e)}")
            # 기본값 반환
            return {
                "analysis": {
                    "face_shape": face_shape,
                    "personal_color": skin_tone,
                    "features": "분석 실패"
                },
                "recommendations": []
            }

    def _merge_recommendations(
        self,
        gemini_recommendations: List[Dict],
        ml_recommendations: List[Dict],
        face_shape: str,
        skin_tone: str
    ) -> List[Dict]:
        """
        Gemini와 ML 추천 결과 병합 (중복 제거)

        Args:
            gemini_recommendations: Gemini 추천 리스트
            ml_recommendations: ML 추천 리스트
            face_shape: 얼굴형
            skin_tone: 피부톤

        Returns:
            병합된 추천 리스트 (최대 7개)
        """
        merged = []
        seen_styles = set()

        # 1. Gemini 추천 추가 (최대 4개)
        for rec in gemini_recommendations:
            style_name = rec.get("style_name", "").strip()

            if not style_name:
                continue

            # 띄어쓰기 정규화 적용 (중복 검사용)
            normalized_name = normalize_style_name(style_name)

            if normalized_name in seen_styles:
                continue

            # hairstyle_id 찾기 (정규화된 이름으로)
            hairstyle_id = None
            if self.ml_available and self.ml_recommender:
                hairstyle_id = self.ml_recommender.style_to_idx.get(normalized_name)

            # ML 점수 추가 (정규화된 이름 사용)
            ml_score = 0.0
            if self.ml_available and self.ml_recommender:
                try:
                    ml_score = self.ml_recommender.predict_score(
                        face_shape, skin_tone, style_name
                    )
                except:
                    pass

            merged.append({
                "hairstyle_id": hairstyle_id,  # ✅ DB ID 추가
                "style_name": style_name,
                "reason": rec.get("reason", ""),
                "source": "gemini",
                "score": ml_score,  # ✅ score로 필드명 통일
                "rank": len(merged) + 1
            })

            seen_styles.add(normalized_name)

        # 2. ML 추천 추가 (중복 제외, 최대 3개)
        for rec in ml_recommendations:
            if len(merged) >= 7:  # 최대 7개
                break

            hairstyle_id = rec.get("hairstyle_id")  # ✅ ML에서 ID 가져오기
            style_name = rec.get("hairstyle", "").strip()
            ml_score = rec.get("score", 0.0)

            if not style_name:
                continue

            # 띄어쓰기 정규화 적용 (중복 검사용)
            normalized_name = normalize_style_name(style_name)

            if normalized_name in seen_styles:
                continue

            # 템플릿 기반 이유 생성
            if self.reason_generator:
                try:
                    reason = self.reason_generator.generate_with_score(
                        face_shape, skin_tone, style_name, ml_score
                    )
                except Exception as e:
                    logger.warning(f"⚠️ 이유 생성 실패: {str(e)}")
                    reason = f"ML 모델 추천 (점수: {ml_score:.1f})"
            else:
                reason = f"ML 모델 추천 (점수: {ml_score:.1f})"

            merged.append({
                "hairstyle_id": hairstyle_id,  # ✅ DB ID 추가
                "style_name": style_name,
                "reason": reason,
                "source": "ml",
                "score": ml_score,  # ✅ score로 필드명 통일
                "rank": len(merged) + 1
            })

            seen_styles.add(normalized_name)

        logger.info(
            f"✅ 추천 병합 완료: Gemini {len(gemini_recommendations)}개 + "
            f"ML {len(ml_recommendations)}개 → 최종 {len(merged)}개"
        )

        return merged

    def recommend(
        self,
        image_data: bytes,
        face_shape: str,
        skin_tone: str
    ) -> Dict:
        """
        하이브리드 추천 실행

        Args:
            image_data: 이미지 바이트
            face_shape: 얼굴형
            skin_tone: 피부톤

        Returns:
            추천 결과 딕셔너리
        """
        logger.info(f"🎨 하이브리드 추천 시작: {face_shape} + {skin_tone}")

        # 1. Gemini 추천 (4개)
        gemini_result = self._call_gemini(image_data, face_shape, skin_tone)
        gemini_recommendations = gemini_result.get("recommendations", [])

        # 2. ML 추천 (Top-3)
        ml_recommendations = []
        if self.ml_available and self.ml_recommender:
            try:
                ml_recommendations = self.ml_recommender.recommend_top_k(
                    face_shape, skin_tone, k=3
                )
            except Exception as e:
                logger.error(f"❌ ML 추천 실패: {str(e)}")

        # 3. 병합 (중복 제거)
        merged_recommendations = self._merge_recommendations(
            gemini_recommendations,
            ml_recommendations,
            face_shape,
            skin_tone
        )

        # 4. 결과 구성
        result = {
            "analysis": gemini_result.get("analysis", {
                "face_shape": face_shape,
                "personal_color": skin_tone,
                "features": "자동 분석"
            }),
            "recommendations": merged_recommendations,
            "meta": {
                "total_count": len(merged_recommendations),
                "gemini_count": len([r for r in merged_recommendations if r["source"] == "gemini"]),
                "ml_count": len([r for r in merged_recommendations if r["source"] == "ml"]),
                "method": "hybrid"
            }
        }

        logger.info(f"✅ 하이브리드 추천 완료: 총 {len(merged_recommendations)}개")

        return result


# ========== 싱글톤 인스턴스 ==========
_hybrid_service_instance = None


def get_hybrid_service(gemini_api_key: str) -> HybridRecommendationService:
    """
    하이브리드 서비스 싱글톤 인스턴스 가져오기

    Args:
        gemini_api_key: Gemini API 키

    Returns:
        HybridRecommendationService 인스턴스
    """
    global _hybrid_service_instance

    if _hybrid_service_instance is None:
        logger.info("🔧 하이브리드 추천 서비스 초기화 중...")
        _hybrid_service_instance = HybridRecommendationService(gemini_api_key)
        logger.info("✅ 하이브리드 추천 서비스 준비 완료")

    return _hybrid_service_instance
