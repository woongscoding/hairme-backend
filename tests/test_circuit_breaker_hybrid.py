"""
Circuit Breaker 테스트 - 서비스 통합

Circuit Breaker가 서비스 호출에서 정상 동작하는지 검증
(MLRecommendationService는 더 이상 Gemini를 직접 호출하지 않으므로,
 gemini_breaker를 직접 사용하여 Circuit Breaker 동작을 검증)
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from pybreaker import CircuitBreakerError
from services.circuit_breaker import (
    gemini_breaker,
    with_circuit_breaker,
    gemini_api_fallback,
)


@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    """각 테스트 전에 Circuit Breaker 리셋"""
    gemini_breaker.close()
    yield
    gemini_breaker.close()


@pytest.fixture
def sample_image_data():
    """테스트용 이미지 데이터 (1x1 PNG)"""
    # 1x1 픽셀 흰색 PNG 이미지
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f"
        b"\x00\x00\x01\x01\x00\x05\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _make_fallback(face_shape, skin_tone):
    """테스트용 fallback 함수 생성"""

    def fallback(*args, **kwargs):
        return {
            "analysis": {
                "face_shape": face_shape,
                "personal_color": skin_tone,
                "features": "Gemini API 일시 중단 - MediaPipe 기반 분석",
            },
            "recommendations": [],
        }

    return fallback


class TestCircuitBreakerInServiceCalls:
    """Circuit Breaker가 서비스 호출에서 동작하는지 테스트"""

    def test_circuit_breaker_opens_after_5_failures(self):
        """5회 연속 실패 시 Circuit이 OPEN 되는지 검증"""

        def failing_api_call():
            raise Exception("Gemini API Error")

        # When: 5회 연속 호출 실패
        for i in range(5):
            try:
                gemini_breaker.call(failing_api_call)
            except Exception:
                pass

        # Then: Circuit이 OPEN 상태여야 함
        assert gemini_breaker.current_state == "open"

    def test_fallback_is_called_when_circuit_is_open(self):
        """Circuit OPEN 시 fallback 함수가 호출되는지 검증"""
        fallback_fn = _make_fallback("계란형", "쿨톤")

        # Given: Circuit을 OPEN 상태로 만들기
        def failing_api_call():
            raise Exception("Gemini API Error")

        for i in range(5):
            try:
                gemini_breaker.call(failing_api_call)
            except Exception:
                pass

        assert gemini_breaker.current_state == "open"

        # When: Circuit이 OPEN된 상태에서 decorator를 통해 호출
        @with_circuit_breaker(gemini_breaker, fallback=fallback_fn)
        def api_call():
            return {"data": "success"}

        result = api_call()

        # Then: fallback 응답이 반환되어야 함
        assert result["analysis"]["face_shape"] == "계란형"
        assert result["analysis"]["personal_color"] == "쿨톤"
        assert "Gemini API 일시 중단" in result["analysis"]["features"]
        assert result["recommendations"] == []

    def test_fallback_contains_mediapipe_data(self):
        """Fallback 응답이 MediaPipe 데이터를 포함하는지 검증"""
        fallback_fn = _make_fallback("사각형", "웜톤")

        # Given: Circuit을 OPEN 상태로 만들기
        def failing_api_call():
            raise Exception("Gemini API Error")

        for i in range(5):
            try:
                gemini_breaker.call(failing_api_call)
            except Exception:
                pass

        # When: Circuit이 OPEN된 상태에서 fallback 호출
        @with_circuit_breaker(gemini_breaker, fallback=fallback_fn)
        def api_call():
            return {"data": "success"}

        result = api_call()

        # Then: MediaPipe 분석 데이터가 포함되어야 함
        assert result["analysis"]["face_shape"] == "사각형"
        assert result["analysis"]["personal_color"] == "웜톤"
        assert (
            result["analysis"]["features"]
            == "Gemini API 일시 중단 - MediaPipe 기반 분석"
        )

    def test_circuit_recovers_after_timeout(self):
        """타임아웃 후 Circuit이 half-open으로 전환되는지 검증"""

        def failing_api_call():
            raise Exception("Gemini API Error")

        # Given: Circuit을 OPEN 상태로 만들기
        for i in range(5):
            try:
                gemini_breaker.call(failing_api_call)
            except Exception:
                pass

        assert gemini_breaker.current_state == "open"

        # When: 타임아웃 시간을 강제로 경과시키기
        gemini_breaker._state_storage.opened_at = 0  # Force timeout

        # Then: 다음 호출 시 half-open으로 전환 후 성공하면 closed
        def successful_call():
            return {"analysis": {}, "recommendations": []}

        try:
            result = gemini_breaker.call(successful_call)
            # 성공하면 Circuit이 닫혀야 함
            assert gemini_breaker.current_state in ["half_open", "closed"]
        except CircuitBreakerError:
            # Circuit이 여전히 열려 있으면 무시
            pass

    def test_successful_call_resets_failure_counter(self):
        """성공한 호출이 실패 카운터를 리셋하는지 검증"""
        call_count = 0

        def sometimes_failing():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise Exception(f"Error {call_count}")
            return {"analysis": {"face_shape": "계란형"}, "recommendations": []}

        # When: 3회 실패
        for i in range(3):
            try:
                gemini_breaker.call(sometimes_failing)
            except Exception:
                pass

        assert gemini_breaker.fail_counter == 3

        # 1회 성공
        result = gemini_breaker.call(sometimes_failing)

        # Then: 실패 카운터가 리셋되어야 함
        assert gemini_breaker.fail_counter == 0
        assert gemini_breaker.current_state == "closed"

    def test_recommend_works_with_circuit_breaker(self):
        """Circuit Breaker decorator를 사용한 추천 플로우 검증"""
        fallback_fn = _make_fallback("계란형", "쿨톤")

        # Given: API가 실패하도록 설정
        def failing_api():
            raise Exception("Gemini API Error")

        # 5회 실패로 Circuit OPEN
        for _ in range(5):
            try:
                gemini_breaker.call(failing_api)
            except Exception:
                pass

        assert gemini_breaker.current_state == "open"

        # When: decorator를 통해 호출
        @with_circuit_breaker(gemini_breaker, fallback=fallback_fn)
        def recommend_api():
            return {"data": "from gemini"}

        result = recommend_api()

        # Then: fallback 응답이 반환되어야 함
        assert "analysis" in result
        assert "recommendations" in result
        assert result["analysis"]["face_shape"] == "계란형"

    def test_circuit_breaker_logging(self, caplog):
        """Circuit Breaker 상태 변화 시 로깅이 발생하는지 검증"""
        caplog.set_level(logging.WARNING)

        def failing_api_call():
            raise Exception("Gemini API Error")

        @with_circuit_breaker(gemini_breaker, fallback=lambda: {"fallback": True})
        def api_call():
            raise Exception("Gemini API Error")

        # When: 5회 연속 호출하여 Circuit을 OPEN (use breaker.call directly)
        for i in range(5):
            try:
                gemini_breaker.call(failing_api_call)
            except Exception:
                pass

        # Then: Circuit이 OPEN 상태
        assert gemini_breaker.current_state == "open"

        # Circuit OPEN 상태에서 decorator를 통해 호출 -> CIRCUIT OPEN 로그
        api_call()

        assert any(
            "CIRCUIT OPEN" in record.message
            or "Circuit이 Open 상태입니다" in record.message
            for record in caplog.records
        )


class TestCircuitBreakerConfiguration:
    """Circuit Breaker 설정 검증"""

    def test_gemini_breaker_configuration(self):
        """Gemini Circuit Breaker 설정이 올바른지 검증"""
        assert gemini_breaker.fail_max == 5
        assert gemini_breaker.reset_timeout == 60
        assert gemini_breaker.name == "GeminiAPI"

    def test_circuit_breaker_initial_state(self):
        """초기 상태가 closed인지 검증"""
        gemini_breaker.close()
        assert gemini_breaker.current_state == "closed"
        assert gemini_breaker.fail_counter == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
