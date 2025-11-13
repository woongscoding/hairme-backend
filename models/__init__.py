# models/__init__.py
"""
HairMe 백엔드 - 얼굴 분석 모듈

이 패키지는 MediaPipe 기반 얼굴 특징 추출 기능을 제공합니다.
"""

# ❌ Haar Cascade 제거됨 (v20.2.0)
# from .face_analyzer import extract_face_features, create_enhanced_prompt, FaceFeatures

# ✅ MediaPipe 사용 (v20.2.0+)
from .mediapipe_analyzer import MediaPipeFaceAnalyzer, MediaPipeFaceFeatures

__all__ = ["MediaPipeFaceAnalyzer", "MediaPipeFaceFeatures"]