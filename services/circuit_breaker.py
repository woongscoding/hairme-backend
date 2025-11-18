"""Circuit Breaker implementation for external API calls"""

from pybreaker import CircuitBreaker, CircuitBreakerError
from functools import wraps
from typing import Callable, Any, Optional
from core.logging import logger


# ========== Circuit Breaker Configuration ==========

# Gemini API Circuit Breaker
# - fail_max=5: Open after 5 consecutive failures
# - reset_timeout=60: Wait 60 seconds before trying again
# - name: For logging and monitoring
gemini_breaker = CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    name='GeminiAPI'
)


class CircuitBreakerListener:
    """Custom listener for circuit breaker events"""

    @staticmethod
    def on_open(circuit_breaker: CircuitBreaker, *args, **kwargs):
        """Called when circuit breaker opens (enters failure state)"""
        logger.error(
            f"[CIRCUIT BREAKER OPEN] {circuit_breaker.name}: "
            f"5회 연속 실패로 인해 Circuit이 Open 되었습니다. "
            f"60초 후 재시도합니다."
        )

    @staticmethod
    def on_close(circuit_breaker: CircuitBreaker, *args, **kwargs):
        """Called when circuit breaker closes (recovers)"""
        logger.info(
            f"[CIRCUIT BREAKER CLOSED] {circuit_breaker.name}: "
            f"Circuit이 정상 복구되었습니다."
        )

    @staticmethod
    def on_half_open(circuit_breaker: CircuitBreaker, *args, **kwargs):
        """Called when circuit breaker enters half-open state (testing recovery)"""
        logger.warning(
            f"[CIRCUIT BREAKER HALF-OPEN] {circuit_breaker.name}: "
            f"복구 테스트를 시작합니다."
        )

    @staticmethod
    def on_failure(circuit_breaker: CircuitBreaker, exception: Exception, *args, **kwargs):
        """Called on each failure"""
        logger.warning(
            f"[CIRCUIT BREAKER FAILURE] {circuit_breaker.name}: "
            f"실패 ({circuit_breaker.fail_counter}/{circuit_breaker.fail_max}): {str(exception)}"
        )

    @staticmethod
    def on_success(circuit_breaker: CircuitBreaker, *args, **kwargs):
        """Called on each success"""
        if circuit_breaker.fail_counter > 0:
            logger.info(
                f"[CIRCUIT BREAKER SUCCESS] {circuit_breaker.name}: "
                f"호출 성공. 실패 카운터 리셋."
            )


# Listener registration disabled due to compatibility issues with current pybreaker version
# Circuit breaker will still function correctly but without detailed event logging


def with_circuit_breaker(
    breaker: CircuitBreaker,
    fallback: Optional[Callable] = None
):
    """
    Decorator to apply circuit breaker to a function

    Args:
        breaker: CircuitBreaker instance to use
        fallback: Optional fallback function to call when circuit is open

    Example:
        @with_circuit_breaker(gemini_breaker, fallback=my_fallback)
        def call_api():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                # Try calling the function through circuit breaker
                return breaker.call(func, *args, **kwargs)

            except CircuitBreakerError as e:
                # Circuit is open - API is down
                logger.error(
                    f"[CIRCUIT OPEN] {breaker.name}: Circuit이 Open 상태입니다. "
                    f"폴백 전략을 사용합니다."
                )

                # Use fallback if provided
                if fallback:
                    logger.info(f"[FALLBACK] {breaker.name}: 폴백 함수 실행")
                    return fallback(*args, **kwargs)

                # Raise custom exception if no fallback
                from core.exceptions import CircuitBreakerOpenException
                raise CircuitBreakerOpenException(
                    service_name=breaker.name,
                    message=f"{breaker.name} Circuit Breaker가 Open 상태입니다. 잠시 후 다시 시도해주세요."
                )

        return wrapper
    return decorator


# ========== Fallback Strategies ==========

def gemini_api_fallback(*args, **kwargs) -> dict:
    """
    Fallback strategy when Gemini API is down

    Returns basic response using only MediaPipe data
    """
    logger.warning("[FALLBACK] Gemini API 사용 불가. MediaPipe 데이터만 사용.")

    # Check if MediaPipe features are available in kwargs
    mp_features = kwargs.get('mp_features')

    if mp_features:
        # Use MediaPipe data as fallback
        return {
            "analysis": {
                "face_shape": mp_features.face_shape,
                "personal_color": mp_features.skin_tone,
                "features": "MediaPipe 기반 분석 (Gemini API 일시 중단)"
            },
            "recommendations": [
                {
                    "style_name": "기본 추천",
                    "reason": "AI 분석 일시 중단. MediaPipe 기반 추천입니다."
                }
            ],
            "fallback": True,
            "fallback_reason": "Gemini API Circuit Breaker OPEN"
        }

    # No MediaPipe data available
    return {
        "analysis": {
            "face_shape": "알 수 없음",
            "personal_color": "알 수 없음",
            "features": "얼굴 분석 일시 불가"
        },
        "recommendations": [
            {
                "style_name": "서비스 일시 중단",
                "reason": "AI 분석 시스템이 일시적으로 중단되었습니다. 잠시 후 다시 시도해주세요."
            }
        ],
        "fallback": True,
        "fallback_reason": "All detection methods unavailable"
    }


def get_circuit_breaker_status() -> dict:
    """
    Get current status of all circuit breakers

    Returns:
        Dictionary with circuit breaker statistics
    """
    return {
        "gemini_api": {
            "state": str(gemini_breaker.current_state),
            "fail_counter": gemini_breaker.fail_counter,
            "fail_max": gemini_breaker.fail_max,
            "reset_timeout": gemini_breaker._reset_timeout,
            "is_open": gemini_breaker.current_state == "open",
            "is_closed": gemini_breaker.current_state == "closed",
            "is_half_open": gemini_breaker.current_state == "half-open"
        }
    }


def reset_circuit_breakers():
    """Reset all circuit breakers (admin function)"""
    gemini_breaker.reset()
    logger.info("[ADMIN] All circuit breakers have been reset")
