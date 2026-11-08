# -*- coding: utf-8 -*-
"""Gemini 이미지 호출의 타임아웃·재시도 분류 테스트 (services/gemini_image_config)

- 클라이언트에 HTTP 타임아웃이 걸린다 (멈춘 호출이 Lambda 타임아웃까지 가지 않게)
- 재시도해도 같은 결과인 실패(프롬프트 차단, 후보 없음, 안전 종료, 4xx)는
  한 번만 호출한다 (비용 최대 3배 방지)
- 일시적 오류(5xx, 429, 타임아웃·연결 오류)와 "이미지 없이 정상 종료"는 재시도한다
- 서비스의 사용자 안내 문구와 반환 dict 모양은 그대로다

실제 Gemini 는 호출하지 않는다 (generate_content 는 목).
"""

import io
import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")

from unittest.mock import MagicMock, patch

import httpx
import pytest
from google.genai import errors, types
from PIL import Image

from services import gemini_image_config as gic
from services.gemini_image_config import (
    FAILURE_BLOCKED,
    FAILURE_CLIENT_ERROR,
    FAILURE_TRANSIENT,
    GEMINI_IMAGE_TIMEOUT_SECONDS,
    build_image_client,
    generate_image,
    is_retryable_error,
)

PNG_RAW = b"\x89PNG\r\n\x1a\nfake-image"

SYNTH_FAIL_MESSAGE = "AI가 잠깐 졸았나봐요.. 다시 한번 눌러주세요!"
COLOR_FAIL_MESSAGE = "염색 시뮬레이션 생성에 실패했습니다. 다시 시도해주세요."


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 20, 30)).save(buf, "PNG")
    return buf.getvalue()


# ----- 실제 SDK 타입으로 만든 응답 (목 객체의 아무 속성이나 참이 되는 문제를 피한다) -----


def _image_response():
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(data=PNG_RAW, mime_type="image/png")
                        )
                    ],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )


def _blocked_prompt_response():
    """프롬프트가 차단되면 후보가 비어 있다 (예전 코드는 [0] 에서 IndexError → 재시도)"""
    return types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(
            block_reason=types.BlockedReason.PROHIBITED_CONTENT
        )
    )


def _empty_candidates_response():
    return types.GenerateContentResponse(candidates=[])


def _safety_finish_response(reason=types.FinishReason.IMAGE_SAFETY):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(finish_reason=reason)]
    )


def _text_only_response():
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model", parts=[types.Part(text="이미지를 만들 수 없어요")]
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )


def _server_error(code=503):
    return errors.ServerError(
        code, {"error": {"code": code, "message": "x", "status": "UNAVAILABLE"}}
    )


def _client_error(code=400):
    return errors.ClientError(
        code, {"error": {"code": code, "message": "x", "status": "INVALID_ARGUMENT"}}
    )


def _client(responses):
    client = MagicMock()
    client.models.generate_content.side_effect = responses
    return client


@pytest.fixture(autouse=True)
def _no_sleep():
    with patch("services.gemini_image_config.time.sleep") as sleep:
        yield sleep


# ========== 클라이언트 타임아웃 ==========


class TestClientTimeout:
    def test_builder_passes_timeout_in_milliseconds(self):
        with patch("google.genai.Client") as client_cls:
            build_image_client(api_key="k")

        http_options = client_cls.call_args.kwargs["http_options"]
        assert http_options.timeout == GEMINI_IMAGE_TIMEOUT_SECONDS * 1000
        # SDK 자체 재시도는 켜지 않는다 (재시도는 generate_image 한 곳에서)
        assert http_options.retry_options is None

    def test_real_client_carries_timeout(self):
        """실제 genai.Client 에 값이 들어간다 (생성만 하고 호출하지 않는다)"""
        client = build_image_client(api_key="test-key")
        assert client._api_client._http_options.timeout == (
            GEMINI_IMAGE_TIMEOUT_SECONDS * 1000
        )

    def test_timeout_leaves_room_for_a_retry_within_lambda(self):
        # Lambda 120초: 타임아웃 1회 + 재시도 1회가 들어가야 한다
        assert GEMINI_IMAGE_TIMEOUT_SECONDS * 2 < 120
        assert gic.GEMINI_IMAGE_DEADLINE_SECONDS < 120

    def test_hairstyle_service_uses_shared_builder(self):
        from services.hairstyle_synthesis_service import HairstyleSynthesisService

        sentinel = object()
        with patch(
            "services.hairstyle_synthesis_service.build_image_client",
            return_value=sentinel,
        ) as builder:
            assert HairstyleSynthesisService().client is sentinel
        builder.assert_called_once()

    def test_hair_color_service_uses_shared_builder(self):
        from services.hair_color_service import HairColorService

        sentinel = object()
        with patch(
            "services.hair_color_service.build_image_client", return_value=sentinel
        ) as builder:
            assert HairColorService().gemini_client is sentinel
        builder.assert_called_once()


# ========== 재시도 분류 (generate_image) ==========


class TestNoRetryOnHopelessFailures:
    @pytest.mark.parametrize(
        "response",
        [
            _blocked_prompt_response(),
            _empty_candidates_response(),
            _safety_finish_response(types.FinishReason.IMAGE_SAFETY),
            _safety_finish_response(types.FinishReason.SAFETY),
            _safety_finish_response(types.FinishReason.PROHIBITED_CONTENT),
            _safety_finish_response(types.FinishReason.IMAGE_PROHIBITED_CONTENT),
            _safety_finish_response(types.FinishReason.BLOCKLIST),
        ],
        ids=[
            "prompt_blocked",
            "empty_candidates",
            "image_safety",
            "safety",
            "prohibited",
            "image_prohibited",
            "blocklist",
        ],
    )
    def test_blocked_response_is_called_once(self, response, _no_sleep):
        client = _client([response, _image_response()])

        result = generate_image(client, ["p"], config=None)

        assert result.success is False
        assert result.failure == FAILURE_BLOCKED
        assert result.api_calls == 1
        assert client.models.generate_content.call_count == 1
        _no_sleep.assert_not_called()

    @pytest.mark.parametrize("code", [400, 401, 403, 404])
    def test_client_error_is_not_retried(self, code):
        client = _client([_client_error(code), _image_response()])

        result = generate_image(client, ["p"], config=None)

        assert result.success is False
        assert result.failure == FAILURE_CLIENT_ERROR
        assert client.models.generate_content.call_count == 1

    def test_unclassified_exception_is_not_retried(self):
        client = _client([ValueError("parse"), _image_response()])

        result = generate_image(client, ["p"], config=None)

        assert result.success is False
        assert client.models.generate_content.call_count == 1


class TestRetryOnTransientFailures:
    @pytest.mark.parametrize(
        "error",
        [
            _server_error(500),
            _server_error(503),
            _client_error(429),
            httpx.ReadTimeout("timed out"),
            httpx.ConnectError("refused"),
            TimeoutError("t"),
        ],
        ids=["500", "503", "429", "read_timeout", "connect_error", "timeout_error"],
    )
    def test_transient_error_then_success(self, error, _no_sleep):
        client = _client([error, _image_response()])

        result = generate_image(client, ["p"], config=None)

        assert result.success is True
        assert result.api_calls == 2
        assert client.models.generate_content.call_count == 2
        _no_sleep.assert_called_once()

    def test_transient_errors_stop_at_max_attempts(self):
        client = _client([_server_error() for _ in range(5)])

        result = generate_image(client, ["p"], config=None)

        assert result.success is False
        assert result.failure == FAILURE_TRANSIENT
        assert result.api_calls == 3

    def test_text_only_response_is_retried_as_before(self):
        """이미지를 빼먹은 정상 종료는 기존처럼 재시도한다"""
        client = _client([_text_only_response(), _image_response()])

        result = generate_image(client, ["p"], config=None)

        assert result.success is True
        assert result.api_calls == 2

    def test_no_retry_when_deadline_would_be_exceeded(self):
        """타임아웃이 한 번 났으면 다음 시도가 Lambda 시간을 넘기므로 접는다"""
        client = _client([httpx.ReadTimeout("t"), _image_response()])
        ticks = iter([0.0, gic.GEMINI_IMAGE_DEADLINE_SECONDS - 10.0])

        result = generate_image(client, ["p"], config=None, clock=lambda: next(ticks))

        assert result.success is False
        assert client.models.generate_content.call_count == 1

    def test_classifier(self):
        assert is_retryable_error(_server_error(500))
        assert is_retryable_error(_client_error(429))
        assert is_retryable_error(httpx.ReadTimeout("t"))
        assert not is_retryable_error(_client_error(400))
        assert not is_retryable_error(IndexError("list index out of range"))


# ========== 서비스 3경로: 문구·반환 모양 유지 + 분류 적용 ==========


def _synthesis_service(responses):
    from services.hairstyle_synthesis_service import HairstyleSynthesisService

    service = HairstyleSynthesisService()
    service._client = _client(responses)
    return service, service._client


def _color_service(responses):
    from services.hair_color_service import HairColorService

    service = HairColorService()
    service._gemini_client = _client(responses)
    return service, service._gemini_client


def _call(kind, responses):
    if kind == "synthesize":
        service, client = _synthesis_service(responses)
        return service.synthesize_hairstyle(_png_bytes(), "투블럭컷"), client
    if kind == "reference":
        service, client = _synthesis_service(responses)
        return service.synthesize_with_reference(_png_bytes(), _png_bytes()), client
    service, client = _color_service(responses)
    return service.synthesize_hair_color(_png_bytes(), "밀크브라운", "#C4A484"), client


KINDS = ["synthesize", "reference", "hair_color"]
FAIL_MESSAGES = {
    "synthesize": SYNTH_FAIL_MESSAGE,
    "reference": SYNTH_FAIL_MESSAGE,
    "hair_color": COLOR_FAIL_MESSAGE,
}


class TestServices:
    @pytest.mark.parametrize("kind", KINDS)
    def test_blocked_prompt_called_once_with_same_message(self, kind):
        result, client = _call(kind, [_blocked_prompt_response(), _image_response()])

        assert client.models.generate_content.call_count == 1
        assert result["success"] is False
        assert result["message"] == FAIL_MESSAGES[kind]
        assert result["image_base64"] is None
        assert result["image_format"] is None
        assert result["api_calls"] == 1

    @pytest.mark.parametrize("kind", KINDS)
    def test_retry_on_5xx(self, kind):
        result, client = _call(kind, [_server_error(503), _image_response()])

        assert client.models.generate_content.call_count == 2
        assert result["success"] is True
        assert result["api_calls"] == 2
        assert result["image_format"] == "png"

    @pytest.mark.parametrize("kind", KINDS)
    def test_client_error_called_once(self, kind):
        result, client = _call(kind, [_client_error(400), _image_response()])

        assert client.models.generate_content.call_count == 1
        assert result["success"] is False
        assert result["message"] == FAIL_MESSAGES[kind]

    @pytest.mark.parametrize("kind", KINDS)
    def test_raw_bytes_are_base64_encoded(self, kind):
        import base64

        result, _client_mock = _call(kind, [_image_response()])

        assert base64.b64decode(result["image_base64"]) == PNG_RAW
