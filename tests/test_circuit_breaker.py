"""Tests for Circuit Breaker implementation"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pybreaker import CircuitBreakerError

from services.circuit_breaker import (
    gemini_breaker,
    gemini_api_fallback,
    get_circuit_breaker_status,
    reset_circuit_breakers,
    with_circuit_breaker
)
from models.mediapipe_analyzer import MediaPipeFaceFeatures


@pytest.fixture(autouse=True)
def reset_breaker():
    """Reset circuit breaker before each test"""
    gemini_breaker.reset()
    yield
    gemini_breaker.reset()


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


class TestCircuitBreaker:
    """Test suite for Circuit Breaker"""

    def test_circuit_breaker_starts_closed(self):
        """Test that circuit breaker starts in closed state"""
        assert gemini_breaker.current_state == "closed"
        assert gemini_breaker.fail_counter == 0

    def test_circuit_breaker_success_keeps_closed(self):
        """Test that successful calls keep circuit closed"""
        def successful_function():
            return "success"

        result = gemini_breaker.call(successful_function)

        assert result == "success"
        assert gemini_breaker.current_state == "closed"
        assert gemini_breaker.fail_counter == 0

    def test_circuit_breaker_opens_after_max_failures(self):
        """Test that circuit opens after max failures"""
        def failing_function():
            raise Exception("API Error")

        # Trigger failures
        for i in range(5):
            try:
                gemini_breaker.call(failing_function)
            except Exception:
                pass

        # Circuit should be open
        assert gemini_breaker.current_state == "open"

        # Next call should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            gemini_breaker.call(failing_function)

    def test_circuit_breaker_fail_counter_increments(self):
        """Test that fail counter increments on errors"""
        def failing_function():
            raise Exception("API Error")

        # First failure
        try:
            gemini_breaker.call(failing_function)
        except Exception:
            pass

        assert gemini_breaker.fail_counter == 1

        # Second failure
        try:
            gemini_breaker.call(failing_function)
        except Exception:
            pass

        assert gemini_breaker.fail_counter == 2

    def test_circuit_breaker_resets_counter_on_success(self):
        """Test that fail counter resets on success"""
        def sometimes_failing():
            if gemini_breaker.fail_counter < 2:
                raise Exception("Temporary error")
            return "success"

        # Trigger 2 failures
        for _ in range(2):
            try:
                gemini_breaker.call(sometimes_failing)
            except Exception:
                pass

        assert gemini_breaker.fail_counter == 2

        # Success resets counter
        result = gemini_breaker.call(sometimes_failing)
        assert result == "success"
        assert gemini_breaker.fail_counter == 0

    def test_get_circuit_breaker_status(self):
        """Test getting circuit breaker status"""
        status = get_circuit_breaker_status()

        assert "gemini_api" in status
        assert status["gemini_api"]["state"] == "closed"
        assert status["gemini_api"]["fail_counter"] == 0
        assert status["gemini_api"]["fail_max"] == 5
        assert status["gemini_api"]["timeout_duration"] == 60
        assert status["gemini_api"]["is_closed"] is True
        assert status["gemini_api"]["is_open"] is False

    def test_reset_circuit_breakers(self):
        """Test resetting circuit breakers"""
        # Trigger some failures
        def failing_function():
            raise Exception("Error")

        try:
            gemini_breaker.call(failing_function)
        except Exception:
            pass

        assert gemini_breaker.fail_counter > 0

        # Reset
        reset_circuit_breakers()

        assert gemini_breaker.fail_counter == 0
        assert gemini_breaker.current_state == "closed"

    def test_gemini_api_fallback_with_mp_features(self, mock_mp_features):
        """Test fallback with MediaPipe features"""
        result = gemini_api_fallback(mp_features=mock_mp_features)

        assert result["analysis"]["face_shape"] == "계란형"
        assert result["analysis"]["personal_color"] == "밝은 톤"
        assert result["fallback"] is True
        assert result["fallback_reason"] == "Gemini API Circuit Breaker OPEN"
        assert len(result["recommendations"]) > 0

    def test_gemini_api_fallback_without_mp_features(self):
        """Test fallback without MediaPipe features"""
        result = gemini_api_fallback()

        assert result["analysis"]["face_shape"] == "알 수 없음"
        assert result["analysis"]["personal_color"] == "알 수 없음"
        assert result["fallback"] is True
        assert result["fallback_reason"] == "All detection methods unavailable"

    def test_with_circuit_breaker_decorator_success(self):
        """Test circuit breaker decorator with successful call"""
        @with_circuit_breaker(gemini_breaker)
        def successful_api_call():
            return {"data": "success"}

        result = successful_api_call()

        assert result == {"data": "success"}
        assert gemini_breaker.current_state == "closed"

    def test_with_circuit_breaker_decorator_with_fallback(self):
        """Test circuit breaker decorator with fallback"""
        def fallback_function():
            return {"fallback": True}

        # Open the circuit
        def failing_function():
            raise Exception("Error")

        for _ in range(5):
            try:
                gemini_breaker.call(failing_function)
            except Exception:
                pass

        # Now use decorator with fallback
        @with_circuit_breaker(gemini_breaker, fallback=fallback_function)
        def api_call():
            return {"data": "success"}

        result = api_call()

        # Should return fallback
        assert result == {"fallback": True}

    def test_with_circuit_breaker_decorator_no_fallback_raises(self):
        """Test circuit breaker decorator without fallback raises error"""
        # Open the circuit
        def failing_function():
            raise Exception("Error")

        for _ in range(5):
            try:
                gemini_breaker.call(failing_function)
            except Exception:
                pass

        # Decorator without fallback
        @with_circuit_breaker(gemini_breaker)
        def api_call():
            return {"data": "success"}

        # Should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            api_call()

    def test_circuit_breaker_configuration(self):
        """Test circuit breaker configuration"""
        assert gemini_breaker.fail_max == 5
        assert gemini_breaker.timeout_duration == 60
        assert gemini_breaker.name == "GeminiAPI"


class TestCircuitBreakerIntegration:
    """Integration tests for Circuit Breaker with GeminiAnalysisService"""

    @patch('services.gemini_analysis_service.genai.GenerativeModel')
    def test_gemini_service_uses_circuit_breaker(self, mock_genai_model):
        """Test that GeminiAnalysisService uses circuit breaker"""
        from services.gemini_analysis_service import GeminiAnalysisService
        from PIL import Image
        import io

        # Create sample image
        img = Image.new('RGB', (100, 100), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        image_data = img_bytes.getvalue()

        # Mock successful response
        mock_model = MagicMock()
        mock_response = MagicMock()
        import json
        mock_response.text = json.dumps({
            "analysis": {
                "face_shape": "계란형",
                "personal_color": "봄웜",
                "features": "테스트"
            },
            "recommendations": []
        })
        mock_model.generate_content.return_value = mock_response
        mock_genai_model.return_value = mock_model

        service = GeminiAnalysisService()

        # Should work normally
        result = service.analyze_with_gemini(image_data)
        assert "analysis" in result

    def test_gemini_service_fallback_when_circuit_open(self, mock_mp_features):
        """Test that GeminiAnalysisService uses fallback when circuit is open"""
        from services.gemini_analysis_service import GeminiAnalysisService
        from PIL import Image
        import io

        # Create sample image
        img = Image.new('RGB', (100, 100), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        image_data = img_bytes.getvalue()

        # Open the circuit by triggering failures
        def failing_function():
            raise Exception("API Error")

        for _ in range(5):
            try:
                gemini_breaker.call(failing_function)
            except Exception:
                pass

        assert gemini_breaker.current_state == "open"

        # Service should use fallback
        service = GeminiAnalysisService()
        result = service.analyze_with_gemini(image_data, mp_features=mock_mp_features)

        assert result["fallback"] is True
        assert result["analysis"]["face_shape"] == "계란형"
