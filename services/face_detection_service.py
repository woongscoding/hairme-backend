"""Face detection service using MediaPipe and Gemini fallback"""

import io
import json
from typing import Dict, Any, Optional
from PIL import Image
import google.generativeai as genai

from config.settings import settings
from core.logging import logger, log_structured
from models.mediapipe_analyzer import MediaPipeFaceAnalyzer, MediaPipeFaceFeatures


class FaceDetectionService:
    """
    Service for face detection with fallback strategy

    Detection flow:
    1. Try MediaPipe (most accurate - 90%+)
    2. Fallback to Gemini if MediaPipe fails
    """

    def __init__(self, mediapipe_analyzer: Optional[MediaPipeFaceAnalyzer] = None):
        """
        Initialize face detection service

        Args:
            mediapipe_analyzer: MediaPipe analyzer instance (optional)
        """
        self.mediapipe_analyzer = mediapipe_analyzer

    def verify_face_with_gemini(self, image_data: bytes) -> Dict[str, Any]:
        """
        Verify face with Gemini when MediaPipe fails

        Args:
            image_data: Image binary data

        Returns:
            Dictionary with face verification results
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            image.thumbnail((256, 256))

            model = genai.GenerativeModel(settings.MODEL_NAME)
            prompt = """이미지에 사람 얼굴이 있나요?

JSON으로만 답변:
{"has_face": true/false, "face_count": 숫자}"""

            response = model.generate_content([prompt, image])
            result = json.loads(response.text.strip())

            return {
                "has_face": result.get("has_face", False),
                "face_count": result.get("face_count", 0),
                "method": "gemini"
            }

        except Exception as e:
            logger.error(f"Gemini 얼굴 검증 실패: {str(e)}")
            return {
                "has_face": False,
                "face_count": 0,
                "method": "gemini",
                "error": str(e)
            }

    def detect_face(self, image_data: bytes) -> Dict[str, Any]:
        """
        Detect face (MediaPipe first, fallback to Gemini)

        Args:
            image_data: Image binary data

        Returns:
            Dictionary with face detection results
            {
                "has_face": bool,
                "face_count": int,
                "method": str ("mediapipe" or "gemini"),
                "features": MediaPipeFaceFeatures (optional)
            }
        """
        # 1st attempt: MediaPipe (most accurate - 90%+)
        if self.mediapipe_analyzer is not None:
            try:
                mp_features = self.mediapipe_analyzer.analyze(image_data)

                if mp_features:
                    log_structured("face_detection", {
                        "method": "mediapipe",
                        "face_count": 1,
                        "success": True,
                        "face_shape": mp_features.face_shape,
                        "skin_tone": mp_features.skin_tone,
                        "confidence": mp_features.confidence
                    })
                    return {
                        "has_face": True,
                        "face_count": 1,
                        "method": "mediapipe",
                        "features": mp_features
                    }

            except Exception as e:
                logger.warning(f"MediaPipe 얼굴 감지 실패: {str(e)}")

        # 2nd attempt: Gemini (final fallback)
        logger.info("MediaPipe 실패, Gemini로 얼굴 검증 시작...")
        gemini_result = self.verify_face_with_gemini(image_data)

        log_structured("face_detection", {
            "method": "gemini",
            "face_count": gemini_result.get("face_count", 0),
            "success": gemini_result.get("has_face", False)
        })

        return gemini_result
