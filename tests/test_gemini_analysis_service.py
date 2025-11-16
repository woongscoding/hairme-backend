"""Tests for Gemini Analysis Service"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from PIL import Image
import io
from fastapi import HTTPException

from services.gemini_analysis_service import GeminiAnalysisService
from models.mediapipe_analyzer import MediaPipeFaceFeatures


@pytest.fixture
def sample_image_data():
    """Generate sample image bytes"""
    img = Image.new('RGB', (100, 100), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    return img_bytes.getvalue()


@pytest.fixture
def mock_mp_features():
    """Mock MediaPipe features"""
    return MediaPipeFaceFeatures(
        face_shape="계란형",
        skin_tone="밝은 톤",
        face_ratio=1.4,
        forehead_width=120.0,
        cheekbone_width=130.0,
        jaw_width=110.0,
        ITA_value=45.0,
        confidence=0.95
    )


@pytest.fixture
def sample_gemini_response():
    """Sample Gemini API response"""
    return {
        "analysis": {
            "face_shape": "계란형",
            "personal_color": "봄웜",
            "features": "이목구비가 조화로움"
        },
        "recommendations": [
            {"style_name": "레이어드 컷", "reason": "얼굴형에 잘 어울림"},
            {"style_name": "웨이브 펌", "reason": "볼륨감을 줌"},
            {"style_name": "단발 컷", "reason": "깔끔한 인상"}
        ]
    }


class TestGeminiAnalysisService:
    """Test suite for GeminiAnalysisService"""

    def test_init_default_retries(self):
        """Test initialization with default retries"""
        service = GeminiAnalysisService()
        assert service.max_retries == 3

    def test_init_custom_retries(self):
        """Test initialization with custom retries"""
        service = GeminiAnalysisService(max_retries=5)
        assert service.max_retries == 5

    def test_build_prompt_with_mediapipe_hints(self, mock_mp_features):
        """Test prompt building with MediaPipe hints"""
        service = GeminiAnalysisService()
        prompt = service._build_prompt_with_mediapipe_hints(mock_mp_features)

        assert "계란형" in prompt
        assert "밝은 톤" in prompt
        assert "1.40" in prompt
        assert "120" in prompt
        assert "45.0" in prompt
        assert "MediaPipe" in prompt
        assert "95%" in prompt

    @patch('services.gemini_analysis_service.genai.GenerativeModel')
    def test_analyze_with_gemini_success(
        self,
        mock_genai_model,
        sample_image_data,
        sample_gemini_response
    ):
        """Test successful Gemini analysis"""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = f'{sample_gemini_response}'

        import json
        mock_response.text = json.dumps(sample_gemini_response)

        mock_model.generate_content.return_value = mock_response
        mock_genai_model.return_value = mock_model

        service = GeminiAnalysisService()
        result = service.analyze_with_gemini(sample_image_data)

        assert "analysis" in result
        assert "recommendations" in result
        assert result["analysis"]["face_shape"] == "계란형"
        assert len(result["recommendations"]) == 3

    @patch('services.gemini_analysis_service.genai.GenerativeModel')
    def test_analyze_with_mediapipe_features(
        self,
        mock_genai_model,
        sample_image_data,
        mock_mp_features,
        sample_gemini_response
    ):
        """Test Gemini analysis with MediaPipe hints"""
        mock_model = MagicMock()
        mock_response = MagicMock()

        import json
        mock_response.text = json.dumps(sample_gemini_response)

        mock_model.generate_content.return_value = mock_response
        mock_genai_model.return_value = mock_model

        service = GeminiAnalysisService()
        result = service.analyze_with_gemini(sample_image_data, mp_features=mock_mp_features)

        assert "analysis" in result
        mock_model.generate_content.assert_called_once()

    @patch('services.gemini_analysis_service.genai.GenerativeModel')
    def test_analyze_strips_markdown_code_blocks(
        self,
        mock_genai_model,
        sample_image_data,
        sample_gemini_response
    ):
        """Test that markdown code blocks are properly stripped"""
        mock_model = MagicMock()
        mock_response = MagicMock()

        import json
        # Wrap in markdown code blocks
        mock_response.text = f'```json\n{json.dumps(sample_gemini_response)}\n```'

        mock_model.generate_content.return_value = mock_response
        mock_genai_model.return_value = mock_model

        service = GeminiAnalysisService()
        result = service.analyze_with_gemini(sample_image_data)

        assert "analysis" in result
        assert result["analysis"]["face_shape"] == "계란형"

    @patch('services.gemini_analysis_service.genai.GenerativeModel')
    def test_analyze_json_decode_error_with_retry(
        self,
        mock_genai_model,
        sample_image_data,
        sample_gemini_response
    ):
        """Test JSON decode error triggers retry"""
        mock_model = MagicMock()
        mock_response = MagicMock()

        import json
        # First call: invalid JSON
        # Second call: valid JSON
        mock_response.text = 'Invalid JSON'
        mock_model.generate_content.side_effect = [
            mock_response,
            MagicMock(text=json.dumps(sample_gemini_response))
        ]
        mock_genai_model.return_value = mock_model

        service = GeminiAnalysisService(max_retries=3)
        result = service.analyze_with_gemini(sample_image_data)

        # Should succeed on retry
        assert "analysis" in result
        assert mock_model.generate_content.call_count == 2

    @patch('services.gemini_analysis_service.genai.GenerativeModel')
    def test_analyze_json_decode_error_exceeds_retries(
        self,
        mock_genai_model,
        sample_image_data
    ):
        """Test JSON decode error raises exception after max retries"""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = 'Invalid JSON'
        mock_model.generate_content.return_value = mock_response
        mock_genai_model.return_value = mock_model

        service = GeminiAnalysisService(max_retries=2)

        with pytest.raises(HTTPException) as exc_info:
            service.analyze_with_gemini(sample_image_data)

        assert exc_info.value.status_code == 500
        assert "파싱 실패" in exc_info.value.detail
        assert "재시도 2회 초과" in exc_info.value.detail

    @patch('services.gemini_analysis_service.genai.GenerativeModel')
    def test_analyze_api_error_with_retry(
        self,
        mock_genai_model,
        sample_image_data,
        sample_gemini_response
    ):
        """Test API error triggers retry"""
        mock_model = MagicMock()

        import json
        # First call: error
        # Second call: success
        mock_model.generate_content.side_effect = [
            Exception("API Error"),
            MagicMock(text=json.dumps(sample_gemini_response))
        ]
        mock_genai_model.return_value = mock_model

        service = GeminiAnalysisService(max_retries=3)
        result = service.analyze_with_gemini(sample_image_data)

        # Should succeed on retry
        assert "analysis" in result
        assert mock_model.generate_content.call_count == 2

    @patch('services.gemini_analysis_service.genai.GenerativeModel')
    def test_analyze_api_error_exceeds_retries(
        self,
        mock_genai_model,
        sample_image_data
    ):
        """Test API error raises exception after max retries"""
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("Persistent API Error")
        mock_genai_model.return_value = mock_model

        service = GeminiAnalysisService(max_retries=2)

        with pytest.raises(HTTPException) as exc_info:
            service.analyze_with_gemini(sample_image_data)

        assert exc_info.value.status_code == 500
        assert "AI 분석 중 오류" in exc_info.value.detail
        assert "재시도 2회 초과" in exc_info.value.detail

    @patch('services.gemini_analysis_service.genai.GenerativeModel')
    def test_analyze_without_mediapipe_features(
        self,
        mock_genai_model,
        sample_image_data,
        sample_gemini_response
    ):
        """Test analysis without MediaPipe features uses basic prompt"""
        mock_model = MagicMock()
        mock_response = MagicMock()

        import json
        mock_response.text = json.dumps(sample_gemini_response)

        mock_model.generate_content.return_value = mock_response
        mock_genai_model.return_value = mock_model

        service = GeminiAnalysisService()
        result = service.analyze_with_gemini(sample_image_data, mp_features=None)

        assert "analysis" in result
        # Should use basic ANALYSIS_PROMPT
        mock_model.generate_content.assert_called_once()
