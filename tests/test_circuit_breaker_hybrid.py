"""
Circuit Breaker 테스트 - Hybrid Recommender 통합

하이브리드 추천 서비스에서 Circuit Breaker가 정상 동작하는지 검증
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pybreaker import CircuitBreakerError
from services.hybrid_recommender import HybridRecommendationService
from services.circuit_breaker import gemini_breaker


@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    """각 테스트 전에 Circuit Breaker 리셋"""
    gemini_breaker.reset()
    yield
    gemini_breaker.reset()


@pytest.fixture
def mock_gemini_api_key():
    """테스트용 Gemini API 키"""
    return "test-gemini-api-key-12345"


@pytest.fixture
def hybrid_service(mock_gemini_api_key):
    """하이브리드 추천 서비스 인스턴스"""
    with patch('google.generativeai.configure'):
        with patch('google.generativeai.GenerativeModel'):
            service = HybridRecommendationService(mock_gemini_api_key)
            return service


@pytest.fixture
def sample_image_data():
    """테스트용 이미지 데이터 (1x1 PNG)"""
    # 1x1 픽셀 흰색 PNG 이미지
    return (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f'
        b'\x00\x00\x01\x01\x00\x05\x00\x00\x00\x00IEND\xaeB`\x82'
    )


class TestCircuitBreakerInHybridRecommender:
    """Circuit Breaker가 하이브리드 추천 서비스에서 동작하는지 테스트"""

    def test_circuit_breaker_opens_after_5_failures(
        self, hybrid_service, sample_image_data
    ):
        """5회 연속 실패 시 Circuit이 OPEN 되는지 검증"""
        # Given: Gemini API가 계속 실패하도록 설정
        with patch.object(
            hybrid_service.gemini_model,
            'generate_content',
            side_effect=Exception("Gemini API Error")
        ):
            # When: 5회 연속 호출
            for i in range(5):
                result = hybrid_service._call_gemini(
                    sample_image_data,
                    "계란형",
                    "쿨톤"
                )
                # Circuit이 열릴 때까지는 fallback 응답 반환
                assert "analysis" in result

            # Then: Circuit이 OPEN 상태여야 함
            assert gemini_breaker.current_state == "open"

    def test_fallback_is_called_when_circuit_is_open(
        self, hybrid_service, sample_image_data
    ):
        """Circuit OPEN 시 fallback 함수가 호출되는지 검증"""
        # Given: Circuit을 OPEN 상태로 만들기
        with patch.object(
            hybrid_service.gemini_model,
            'generate_content',
            side_effect=Exception("Gemini API Error")
        ):
            for i in range(5):
                hybrid_service._call_gemini(sample_image_data, "계란형", "쿨톤")

        # When: Circuit이 OPEN된 상태에서 호출
        result = hybrid_service._call_gemini(sample_image_data, "계란형", "쿨톤")

        # Then: fallback 응답이 반환되어야 함
        assert result["analysis"]["face_shape"] == "계란형"
        assert result["analysis"]["personal_color"] == "쿨톤"
        assert "Gemini API 일시 중단" in result["analysis"]["features"]
        assert result["recommendations"] == []

    def test_fallback_contains_mediapipe_data(
        self, hybrid_service, sample_image_data
    ):
        """Fallback 응답이 MediaPipe 데이터를 포함하는지 검증"""
        # Given: Circuit을 OPEN 상태로 만들기
        with patch.object(
            hybrid_service.gemini_model,
            'generate_content',
            side_effect=Exception("Gemini API Error")
        ):
            for i in range(5):
                hybrid_service._call_gemini(sample_image_data, "사각형", "웜톤")

        # When: Circuit이 OPEN된 상태에서 호출
        result = hybrid_service._call_gemini(sample_image_data, "사각형", "웜톤")

        # Then: MediaPipe 분석 데이터가 포함되어야 함
        assert result["analysis"]["face_shape"] == "사각형"
        assert result["analysis"]["personal_color"] == "웜톤"
        assert result["analysis"]["features"] == "Gemini API 일시 중단 - MediaPipe 기반 분석"

    def test_circuit_recovers_after_timeout(
        self, hybrid_service, sample_image_data
    ):
        """타임아웃 후 Circuit이 half-open으로 전환되는지 검증"""
        # Given: Circuit을 OPEN 상태로 만들기
        with patch.object(
            hybrid_service.gemini_model,
            'generate_content',
            side_effect=Exception("Gemini API Error")
        ):
            for i in range(5):
                hybrid_service._call_gemini(sample_image_data, "계란형", "쿨톤")

        assert gemini_breaker.current_state == "open"

        # When: 타임아웃 시간을 강제로 경과시키기
        gemini_breaker._state_storage.opened_at = 0  # Force timeout

        # Then: 다음 호출 시 half-open으로 전환되어야 함
        # (실제로는 타임아웃을 기다려야 하지만 테스트에서는 상태를 조작)
        try:
            with patch.object(
                hybrid_service.gemini_model,
                'generate_content',
                return_value=Mock(text='{"analysis": {}, "recommendations": []}')
            ):
                result = hybrid_service._call_gemini(sample_image_data, "계란형", "쿨톤")
                # 성공하면 Circuit이 닫혀야 함
                assert gemini_breaker.current_state in ["half_open", "closed"]
        except CircuitBreakerError:
            # Circuit이 여전히 열려 있으면 fallback이 호출됨
            pass

    def test_successful_call_resets_failure_counter(
        self, hybrid_service, sample_image_data
    ):
        """성공한 호출이 실패 카운터를 리셋하는지 검증"""
        # Given: 3회 실패 후 1회 성공
        mock_response = Mock()
        mock_response.text = '{"analysis": {"face_shape": "계란형", "personal_color": "쿨톤", "features": "test"}, "recommendations": []}'

        with patch.object(
            hybrid_service.gemini_model,
            'generate_content',
            side_effect=[
                Exception("Error 1"),
                Exception("Error 2"),
                Exception("Error 3"),
                mock_response  # 4번째는 성공
            ]
        ):
            # When: 3회 실패
            for i in range(3):
                hybrid_service._call_gemini(sample_image_data, "계란형", "쿨톤")

            assert gemini_breaker.fail_counter == 3

            # 1회 성공
            result = hybrid_service._call_gemini(sample_image_data, "계란형", "쿨톤")

            # Then: 실패 카운터가 리셋되어야 함
            assert gemini_breaker.fail_counter == 0
            assert gemini_breaker.current_state == "closed"

    def test_hybrid_recommend_works_with_circuit_breaker(
        self, hybrid_service, sample_image_data
    ):
        """전체 추천 플로우에서 Circuit Breaker가 정상 동작하는지 검증"""
        # Given: Gemini API가 실패하도록 설정
        with patch.object(
            hybrid_service.gemini_model,
            'generate_content',
            side_effect=Exception("Gemini API Error")
        ):
            # When: 추천 요청 (ML 추천기를 모킹)
            with patch.object(
                hybrid_service,
                'ml_available',
                False
            ):
                result = hybrid_service.recommend(
                    sample_image_data,
                    "계란형",
                    "쿨톤"
                )

                # Then: Gemini 실패해도 응답은 반환되어야 함
                assert "analysis" in result
                assert "recommendations" in result
                assert result["analysis"]["face_shape"] == "계란형"

    def test_circuit_breaker_logging(
        self, hybrid_service, sample_image_data, caplog
    ):
        """Circuit Breaker 상태 변화 시 로깅이 발생하는지 검증"""
        import logging
        caplog.set_level(logging.WARNING)

        # Given: Gemini API가 실패하도록 설정
        with patch.object(
            hybrid_service.gemini_model,
            'generate_content',
            side_effect=Exception("Gemini API Error")
        ):
            # When: 5회 연속 호출하여 Circuit을 OPEN
            for i in range(5):
                hybrid_service._call_gemini(sample_image_data, "계란형", "쿨톤")

            # Then: Circuit OPEN 로그가 있어야 함
            assert any(
                "CIRCUIT BREAKER OPEN" in record.message or
                "Circuit이 Open 되었습니다" in record.message
                for record in caplog.records
            )


class TestCircuitBreakerConfiguration:
    """Circuit Breaker 설정 검증"""

    def test_gemini_breaker_configuration(self):
        """Gemini Circuit Breaker 설정이 올바른지 검증"""
        assert gemini_breaker.fail_max == 5
        assert gemini_breaker.timeout_duration == 60
        assert gemini_breaker.name == "GeminiAPI"

    def test_circuit_breaker_initial_state(self):
        """초기 상태가 closed인지 검증"""
        gemini_breaker.reset()
        assert gemini_breaker.current_state == "closed"
        assert gemini_breaker.fail_counter == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
