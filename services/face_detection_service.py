"""Face detection service using MediaPipe"""

from typing import Dict, Any, Optional

from core.logging import logger, log_structured
from models.mediapipe_analyzer import MediaPipeFaceAnalyzer, MediaPipeFaceFeatures


class FaceDetectionService:
    """
    Service for face detection (MediaPipe only)

    MediaPipe 가 얼굴을 못 찾으면 그대로 "얼굴 없음"을 돌려준다.
    예전에는 Gemini 로 한 번 더 확인했지만, Gemini 는 얼굴 유무만 답하고 특징값을
    주지 않아 "얼굴 있음"이어도 분석을 이어갈 수 없었다(유료 호출 + 500 응답).
    엔드포인트는 has_face=False 를 받으면 NoFaceDetectedException(400) 으로 응답한다.
    """

    def __init__(self, mediapipe_analyzer: Optional[MediaPipeFaceAnalyzer] = None):
        """
        Initialize face detection service

        Args:
            mediapipe_analyzer: MediaPipe analyzer instance (optional)
        """
        self.mediapipe_analyzer = mediapipe_analyzer

    @staticmethod
    def _no_face_result() -> Dict[str, Any]:
        return {
            "has_face": False,
            "face_count": 0,
            "method": "mediapipe",
            "features": None,
        }

    def detect_face(self, image_data: bytes) -> Dict[str, Any]:
        """
        Detect face with MediaPipe

        Args:
            image_data: Image binary data

        Returns:
            Dictionary with face detection results
            {
                "has_face": bool,
                "face_count": int,
                "method": "mediapipe",
                "features": MediaPipeFaceFeatures or None
            }

        Raises:
            RuntimeError: MediaPipe 분석기가 초기화되지 않은 경우 (서버 설정 오류)
        """
        if self.mediapipe_analyzer is None:
            # 서버 설정 오류를 "얼굴 없음"으로 감추지 않는다 (사용자가 재촬영만 반복하게 됨)
            raise RuntimeError("MediaPipe 분석기가 초기화되지 않았습니다")

        mp_features: Optional[MediaPipeFaceFeatures] = None
        try:
            mp_features = self.mediapipe_analyzer.analyze(image_data)
        except Exception as e:
            # 실제 분석기는 내부 예외를 잡아 None 을 돌려주므로 같은 취급을 한다
            logger.error(f"MediaPipe 얼굴 감지 중 예외: {str(e)}", exc_info=True)

        if not mp_features:
            log_structured(
                "face_detection",
                {"method": "mediapipe", "face_count": 0, "success": False},
            )
            return self._no_face_result()

        log_structured(
            "face_detection",
            {
                "method": "mediapipe",
                "face_count": 1,
                "success": True,
                "face_shape": mp_features.face_shape,
                "skin_tone": mp_features.skin_tone,
                "confidence": mp_features.confidence,
            },
        )
        return {
            "has_face": True,
            "face_count": 1,
            "method": "mediapipe",
            "features": mp_features,
        }
