"""Hairstyle synthesis service using Gemini 2.5 Flash Image API"""

import base64
import io
from typing import Dict, Any, Optional
from PIL import Image

from core.logging import logger


class HairstyleSynthesisService:
    """
    Service for synthesizing hairstyles on user photos using Gemini 2.5 Flash Image

    Features:
    - Takes user photo and target hairstyle
    - Generates new image with the hairstyle applied
    - Supports Korean hairstyle names
    - State-of-the-art image generation with multimodal reasoning
    """

    # Model for image generation - Gemini 2.5 Flash Image (nano-banana)
    IMAGE_MODEL = "gemini-2.5-flash-image"

    def __init__(self):
        """Initialize the synthesis service"""
        self._client = None

    @property
    def client(self):
        """Lazy load the Gemini client"""
        if self._client is None:
            from google import genai
            from config.settings import settings
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return self._client

    def synthesize_hairstyle(
        self,
        image_data: bytes,
        hairstyle_name: str,
        gender: str = "male",
        additional_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synthesize a hairstyle on the user's photo

        Args:
            image_data: Original user photo as bytes
            hairstyle_name: Target hairstyle name (e.g., "투블럭컷", "레이어드컷")
            gender: User's gender ("male" or "female")
            additional_instructions: Extra styling instructions (optional)

        Returns:
            Dictionary with:
            - success: bool
            - image_base64: Base64 encoded result image
            - image_format: Image format (e.g., "png")
            - message: Status message
        """
        try:
            from google.genai import types

            # Open the original image
            original_image = Image.open(io.BytesIO(image_data))

            # Build the prompt
            gender_kr = "남성" if gender == "male" else "여성"

            prompt = f"""이 사진의 인물에게 '{hairstyle_name}' 헤어스타일을 적용해주세요.

요구사항:
1. 인물의 얼굴은 그대로 유지하고, 머리카락만 '{hairstyle_name}' 스타일로 자연스럽게 변경해주세요.
2. {gender_kr}용 헤어스타일입니다.
3. 머리카락의 볼륨, 결, 그림자가 자연스럽게 보이도록 해주세요.
4. 배경은 원본과 최대한 비슷하게 유지해주세요.
5. 고품질의 사실적인 이미지를 생성해주세요."""

            if additional_instructions:
                prompt += f"\n6. 추가 요청: {additional_instructions}"

            logger.info(f"🎨 헤어스타일 합성 시작: {hairstyle_name} ({gender})")

            # Call Gemini 3 Pro Image API
            response = self.client.models.generate_content(
                model=self.IMAGE_MODEL,
                contents=[prompt, original_image],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                )
            )

            # Extract the generated image
            result_image = None
            result_text = None

            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    # Image data
                    result_image = part.inline_data
                elif hasattr(part, 'text') and part.text:
                    # Text response
                    result_text = part.text

            if result_image is None:
                logger.error("❌ Gemini가 이미지를 생성하지 않았습니다")
                return {
                    "success": False,
                    "image_base64": None,
                    "image_format": None,
                    "message": "이미지 생성에 실패했습니다. 다시 시도해주세요.",
                    "gemini_response": result_text
                }

            # Convert to base64
            image_bytes = result_image.data

            # 🔍 디버깅: Gemini 응답 데이터 타입 확인
            logger.info(f"📦 Gemini 응답 데이터 타입: {type(image_bytes)}")
            logger.info(f"📦 Gemini 응답 데이터 크기: {len(image_bytes) if image_bytes else 0}")

            # Gemini API가 이미 Base64 문자열을 반환하는 경우 처리
            if isinstance(image_bytes, str):
                # 이미 Base64 문자열인 경우 - 그대로 사용
                logger.info("📦 Gemini가 이미 Base64 문자열을 반환함 - 그대로 사용")
                image_base64 = image_bytes
            elif isinstance(image_bytes, bytes):
                # bytes인 경우 - PNG 시그니처 확인
                if image_bytes[:4] == b'\x89PNG' or image_bytes[:3] == b'\xff\xd8\xff':
                    # 실제 이미지 바이너리 - Base64 인코딩 필요
                    logger.info("📦 실제 이미지 바이너리 - Base64 인코딩 수행")
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                else:
                    # 이미 Base64 인코딩된 bytes일 수 있음 - 디코드해서 확인
                    try:
                        decoded_str = image_bytes.decode('utf-8')
                        # Base64 문자열인지 확인 (iVBOR은 PNG, /9j/는 JPEG의 Base64 시작)
                        if decoded_str.startswith('iVBOR') or decoded_str.startswith('/9j/'):
                            logger.info("📦 이미 Base64 인코딩된 bytes - 그대로 사용")
                            image_base64 = decoded_str
                        else:
                            # 알 수 없는 형식 - 일단 Base64 인코딩
                            logger.info("📦 알 수 없는 형식 - Base64 인코딩 수행")
                            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    except UnicodeDecodeError:
                        # UTF-8 디코드 실패 - 바이너리 데이터로 취급
                        logger.info("📦 바이너리 데이터 - Base64 인코딩 수행")
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            else:
                # 기타 타입 - 일단 bytes로 변환 시도
                logger.warning(f"📦 예상치 못한 타입: {type(image_bytes)}")
                image_base64 = base64.b64encode(bytes(image_bytes)).decode('utf-8')

            # Determine format from mime type
            mime_type = result_image.mime_type  # e.g., "image/png"
            image_format = mime_type.split('/')[-1] if mime_type else "png"

            logger.info(f"✅ 헤어스타일 합성 성공: {hairstyle_name}")

            return {
                "success": True,
                "image_base64": image_base64,
                "image_format": image_format,
                "message": f"'{hairstyle_name}' 스타일이 적용되었습니다.",
                "gemini_response": result_text
            }

        except Exception as e:
            logger.error(f"❌ 헤어스타일 합성 실패: {str(e)}")
            import traceback
            traceback.print_exc()

            return {
                "success": False,
                "image_base64": None,
                "image_format": None,
                "message": f"합성 중 오류가 발생했습니다: {str(e)}",
                "gemini_response": None
            }

    def synthesize_with_reference(
        self,
        user_image_data: bytes,
        reference_image_data: bytes,
        gender: str = "male"
    ) -> Dict[str, Any]:
        """
        Synthesize hairstyle using a reference image

        Args:
            user_image_data: User's photo as bytes
            reference_image_data: Reference hairstyle image as bytes
            gender: User's gender

        Returns:
            Same as synthesize_hairstyle()
        """
        try:
            from google.genai import types

            user_image = Image.open(io.BytesIO(user_image_data))
            reference_image = Image.open(io.BytesIO(reference_image_data))

            gender_kr = "남성" if gender == "male" else "여성"

            prompt = f"""첫 번째 사진의 인물에게 두 번째 사진의 헤어스타일을 적용해주세요.

요구사항:
1. 첫 번째 사진 인물의 얼굴은 그대로 유지합니다.
2. 두 번째 사진의 헤어스타일(길이, 스타일, 볼륨)을 복사합니다.
3. {gender_kr}에게 어울리도록 자연스럽게 적용해주세요.
4. 머리카락 색상은 첫 번째 사진의 원래 색상을 유지해주세요.
5. 고품질의 사실적인 이미지를 생성해주세요."""

            logger.info(f"🎨 레퍼런스 기반 헤어스타일 합성 시작 ({gender})")

            response = self.client.models.generate_content(
                model=self.IMAGE_MODEL,
                contents=[prompt, user_image, reference_image],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                )
            )

            # Extract result
            result_image = None
            result_text = None

            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    result_image = part.inline_data
                elif hasattr(part, 'text') and part.text:
                    result_text = part.text

            if result_image is None:
                return {
                    "success": False,
                    "image_base64": None,
                    "image_format": None,
                    "message": "이미지 생성에 실패했습니다.",
                    "gemini_response": result_text
                }

            image_bytes = result_image.data

            # Gemini API가 이미 Base64 문자열을 반환하는 경우 처리
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

            logger.info(f"✅ 레퍼런스 기반 합성 성공")

            return {
                "success": True,
                "image_base64": image_base64,
                "image_format": image_format,
                "message": "레퍼런스 스타일이 적용되었습니다.",
                "gemini_response": result_text
            }

        except Exception as e:
            logger.error(f"❌ 레퍼런스 기반 합성 실패: {str(e)}")
            return {
                "success": False,
                "image_base64": None,
                "image_format": None,
                "message": f"합성 중 오류가 발생했습니다: {str(e)}",
                "gemini_response": None
            }


# Singleton instance
_synthesis_service: Optional[HairstyleSynthesisService] = None


def get_synthesis_service() -> HairstyleSynthesisService:
    """Get or create the synthesis service singleton"""
    global _synthesis_service
    if _synthesis_service is None:
        _synthesis_service = HairstyleSynthesisService()
    return _synthesis_service
