"""분당 요청 제한(429) 응답 - 한국어 안내

slowapi 기본 핸들러는 {"error": "Rate limit exceeded: 5 per 1 minute"} 를 내려준다.
앱은 402 가 아닌 응답의 본문을 그대로 화면에 붙이므로 영어 문장과 제한 수치가
사용자에게 노출된다. 여기서 다른 거절 응답(core.quota, core.synthesis_lock)과
같은 모양(error 코드 + 한국어 message)으로 바꾼다.

제한 수치("5 per 1 minute")는 본문에 넣지 않는다 - 막힌 사용자가 할 수 있는
행동은 "잠시 뒤 다시" 뿐이고, 남은 호출 수는 우회 시도에 쓰일 수 있다.
원래 문구는 로그에만 남긴다.
"""

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from slowapi.errors import RateLimitExceeded

from core.logging import logger


def rate_limit_response_body() -> dict:
    """429 응답 본문 (앱이 문구를 그대로 표시한다)"""
    return {
        "error": "rate_limit_exceeded",
        "message": "요청이 너무 빠릅니다. 잠시 후 다시 시도해주세요.",
    }


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """slowapi 기본 핸들러(_rate_limit_exceeded_handler) 대체.

    기본 핸들러와 동일하게 제한 헤더를 주입하되, 본문만 한국어로 바꾼다.
    """
    logger.warning(f"요청 제한 초과: path={request.url.path}, limit={exc.detail}")

    response: Response = JSONResponse(
        status_code=429, content=rate_limit_response_body()
    )

    # 헤더 주입은 limiter(headers_enabled)에 달려 있고, 예외가 제한 검사 밖에서
    # 올라온 경우 view_rate_limit 이 없다. 안내 문구가 헤더 때문에 깨지지 않도록
    # 둘 다 있을 때만 주입한다.
    limiter = getattr(request.app.state, "limiter", None)
    view_rate_limit = getattr(request.state, "view_rate_limit", None)
    if limiter is not None and view_rate_limit is not None:
        response = limiter._inject_headers(response, view_rate_limit)

    return response
