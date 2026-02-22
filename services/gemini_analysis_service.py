"""Gemini AI analysis service for face shape and personal color"""

import io
import json
from typing import Dict, Any, Optional
from PIL import Image

# import google.generativeai as genai  # Lazy loaded
from fastapi import HTTPException
from pybreaker import CircuitBreakerError

# from config.settings import settings  # Lazy loaded inside methods
from core.logging import logger
from models.mediapipe_analyzer import MediaPipeFaceFeatures
from services.circuit_breaker import gemini_breaker, gemini_api_fallback

# ========== Gemini Prompts ==========
ANALYSIS_PROMPT = """분석하고 JSON으로 응답:

얼굴형: 계란형/둥근형/각진형/긴형 중 1개
퍼스널컬러: 봄웜/가을웜/여름쿨/겨울쿨 중 1개
헤어스타일 추천 3개 (각 이름 15자, 이유 30자 이내)

JSON 형식:
{
  "analysis": {
    "face_shape": "계란형",
    "personal_color": "봄웜",
    "features": "이목구비 특징"
  },
  "recommendations": [
    {"style_name": "스타일명", "reason": "추천 이유"}
  ]
}"""


class GeminiAnalysisService:
    """
    Service for AI-powered face analysis using Gemini Vision API

    Features:
    - Face shape detection
    - Personal color analysis
    - Hairstyle recommendations
    - MediaPipe hints for improved accuracy
    """

    def __init__(self, max_retries: int = 3):
        """
        Initialize Gemini analysis service

        Args:
            max_retries: Maximum number of retries for API calls
        """
        self.max_retries = max_retries

    def _build_prompt_with_mediapipe_hints(
        self, mp_features: MediaPipeFaceFeatures
    ) -> str:
        """
        Build Gemini prompt with MediaPipe measurement hints

        Args:
            mp_features: MediaPipe analysis results

        Returns:
            Prompt string with MediaPipe data
        """
        return f"""다음 얼굴 사진을 분석하고 JSON으로 응답해주세요.

🔍 **MediaPipe 측정 데이터** (수학적 얼굴 분석 - 신뢰도 {mp_features.confidence:.0%}):
- 얼굴형: {mp_features.face_shape}
- 피부톤: {mp_features.skin_tone}
- 얼굴 비율(높이/너비): {mp_features.face_ratio:.2f}
- 이마 너비: {mp_features.forehead_width:.0f}px
- 광대 너비: {mp_features.cheekbone_width:.0f}px
- 턱 너비: {mp_features.jaw_width:.0f}px
- ITA 값: {mp_features.ITA_value:.1f}°

⚠️ **중요**: 위 MediaPipe 측정값은 수학적으로 계산된 정확한 데이터입니다.
시각적으로 명백히 다르지 않다면 MediaPipe 결과를 그대로 사용하세요.
(참고: 최종 결과는 MediaPipe 값이 우선 채택되므로, 일관성을 위해 같은 값 사용 권장)

**분석 항목:**
1. 얼굴형: 계란형/둥근형/각진형/긴형/하트형 중 1개
2. 퍼스널컬러: 봄웜/가을웜/여름쿨/겨울쿨 중 1개
3. 헤어스타일 추천 3개 (각 이름 15자, 이유 30자 이내)

**JSON 형식:**
{{
  "analysis": {{
    "face_shape": "계란형",
    "personal_color": "봄웜",
    "features": "이목구비 특징 설명"
  }},
  "recommendations": [
    {{"style_name": "스타일명", "reason": "추천 이유"}}
  ]
}}"""

    def _analyze_with_gemini_internal(
        self,
        image_data: bytes,
        mp_features: Optional[MediaPipeFaceFeatures] = None,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        """
        Internal method for Gemini API call (wrapped by circuit breaker)

        Args:
            image_data: Image binary data
            mp_features: MediaPipe analysis results (optional)
            retry_count: Current retry attempt number

        Returns:
            Dictionary with analysis results

        Raises:
            HTTPException: If analysis fails after all retries
        """
        try:
            image = Image.open(io.BytesIO(image_data))

            # Build prompt with or without MediaPipe hints
            if mp_features:
                prompt = self._build_prompt_with_mediapipe_hints(mp_features)
                logger.info(
                    f"✅ MediaPipe 힌트 적용: {mp_features.face_shape} / "
                    f"{mp_features.skin_tone}"
                )
            else:
                prompt = ANALYSIS_PROMPT
                logger.warning("⚠️ MediaPipe 특징 없음, 기본 프롬프트 사용")

            # Call Gemini API
            import google.generativeai as genai
            from config.settings import settings

            model = genai.GenerativeModel(settings.MODEL_NAME)

            # Use temperature=0 for consistent responses
            generation_config = genai.types.GenerationConfig(
                temperature=0.0,
            )

            response = model.generate_content(
                [prompt, image], generation_config=generation_config
            )

            # Parse JSON response
            raw_text = response.text.strip()

            # Clean up markdown code blocks
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            result = json.loads(raw_text.strip())

            logger.info(
                f"✅ Gemini 분석 성공: "
                f"{result.get('analysis', {}).get('face_shape')}"
            )
            return result

        except json.JSONDecodeError as e:
            error_msg = f"JSON 파싱 실패: {str(e)}\n응답 내용: {response.text[:200]}"
            logger.error(error_msg)

            # Retry logic
            if retry_count < self.max_retries:
                logger.warning(
                    f"⚠️ JSON 파싱 실패, 재시도 {retry_count + 1}/{self.max_retries}"
                )
                return self._analyze_with_gemini_internal(
                    image_data, mp_features, retry_count + 1
                )

            raise HTTPException(
                status_code=500,
                detail=f"AI 응답 파싱 실패 (재시도 {self.max_retries}회 초과): {str(e)}",
            )

        except Exception as e:
            error_msg = f"Gemini 분석 실패: {str(e)}"
            logger.error(error_msg)

            # Retry logic for API errors
            if retry_count < self.max_retries:
                logger.warning(
                    f"⚠️ Gemini API 오류, 재시도 {retry_count + 1}/{self.max_retries}"
                )
                return self._analyze_with_gemini_internal(
                    image_data, mp_features, retry_count + 1
                )

            raise HTTPException(
                status_code=500,
                detail=f"AI 분석 중 오류가 발생했습니다 (재시도 {self.max_retries}회 초과): {str(e)}",
            )

    def analyze_with_gemini(
        self,
        image_data: bytes,
        mp_features: Optional[MediaPipeFaceFeatures] = None,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        """
        Analyze face with Gemini Vision API (with Circuit Breaker protection)

        Args:
            image_data: Image binary data
            mp_features: MediaPipe analysis results (optional)
            retry_count: Current retry attempt number

        Returns:
            Dictionary with analysis results
            If circuit is open, returns fallback response

        Raises:
            HTTPException: If analysis fails after all retries
        """
        try:
            # Call through circuit breaker
            return gemini_breaker.call(
                self._analyze_with_gemini_internal, image_data, mp_features, retry_count
            )

        except CircuitBreakerError:
            # Circuit is open - use fallback
            logger.error(
                "[CIRCUIT BREAKER] Gemini API Circuit이 Open 상태입니다. "
                "폴백 응답을 반환합니다."
            )

            return gemini_api_fallback(mp_features=mp_features)
