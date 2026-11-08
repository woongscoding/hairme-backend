# -*- coding: utf-8 -*-
"""Beauty Lambda(main_beauty) 의 분당 요청 제한 429 도 한국어 본문이어야 한다

염색 합성은 Beauty Lambda 에서 돈다. main.py 만 한국어 핸들러(core.rate_limit)를
등록하고 main_beauty.py 는 slowapi 기본 핸들러(영어 + "5 per 1 minute")를 쓰고 있어
같은 앱 화면에 Lambda 에 따라 다른 문구가 나왔다.
"""

import json
import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

from unittest.mock import MagicMock

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request

import main
import main_beauty
from core.rate_limit import rate_limit_exceeded_handler, rate_limit_response_body


def _blocked_response(app):
    """해당 앱에 등록된 429 핸들러가 내려주는 응답 (분당 리미터를 소진시키지 않는다)"""
    handler = app.exception_handlers[RateLimitExceeded]
    request = Request(
        {
            "type": "http",
            "app": app,
            "method": "POST",
            "path": "/api/hair-color/synthesize",
            "headers": [],
            "query_string": b"",
        }
    )
    limit = MagicMock(error_message=None, limit="5 per 1 minute")
    return handler(request, RateLimitExceeded(limit))


def test_beauty_app_registers_the_korean_handler():
    handler = main_beauty.app.exception_handlers[RateLimitExceeded]
    assert handler is rate_limit_exceeded_handler
    assert handler is not _rate_limit_exceeded_handler
    assert handler is main.app.exception_handlers[RateLimitExceeded]


def test_beauty_429_body_matches_main_app():
    beauty = _blocked_response(main_beauty.app)
    main_app = _blocked_response(main.app)

    assert beauty.status_code == main_app.status_code == 429
    beauty_body = json.loads(beauty.body)
    assert beauty_body == json.loads(main_app.body) == rate_limit_response_body()
    # 앱이 본문을 그대로 보여주므로 영어 문장·제한 수치가 새면 안 된다
    text = beauty.body.decode("utf-8")
    assert "Rate limit exceeded" not in text
    assert "per 1 minute" not in text
