"""Tests for face analysis endpoints"""

import sys

from unittest.mock import Mock, patch, MagicMock

from main import app
from core.dependencies import get_face_detection_service, get_hybrid_service


class TestNoFaceWithRealDetectionService:
    """실제 FaceDetectionService + 얼굴 못 찾는 분석기 → 400 (Gemini 호출·500 없음)"""

    def setup_method(self):
        from services.face_detection_service import FaceDetectionService

        analyzer = Mock()
        analyzer.analyze.return_value = None
        self.detector = FaceDetectionService(mediapipe_analyzer=analyzer)
        app.dependency_overrides[get_face_detection_service] = lambda: self.detector
        app.dependency_overrides[get_hybrid_service] = lambda: MagicMock()

    def teardown_method(self):
        app.dependency_overrides.pop(get_face_detection_service, None)
        app.dependency_overrides.pop(get_hybrid_service, None)

    def test_v2_no_face_is_400_without_gemini(self, client, sample_image_bytes):
        fake_genai = MagicMock()
        with patch.dict(sys.modules, {"google.generativeai": fake_genai}):
            response = client.post(
                "/api/v2/analyze-hybrid",
                files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
                data={"gender": "male"},
            )

        assert response.status_code == 400
        body = response.json()
        assert body["success"] is False
        assert body["error"] == "nofacedetected"
        fake_genai.GenerativeModel.assert_not_called()
