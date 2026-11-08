"""Gemini 이미지 생성(헤어 합성·염색 합성) 요청 설정

모델마다 지원하는 옵션이 달라서 한곳에서 맞춘다.

- Nano Banana 2.1 (gemini-nano-banana-*)
  * 출력 512 를 지원하지 않는다 → 1K 로 받고, 무료 등급은 finalize 단계에서 512 로 줄인다
  * 기본 생각 수준이 medium 이라 느리다(최대 27초 측정) → GEMINI_IMAGE_THINKING_LEVEL(minimal)
- gemini-3.1-flash-image 등 이전 모델
  * 512 를 그대로 요청하고 생각 설정은 넣지 않는다 (지원하지 않을 수 있음)

그래서 GEMINI_IMAGE_MODEL 하나만 이전 모델로 바꿔도 그대로 롤백된다.

클라이언트·호출·재시도도 여기서 한 번에 정한다 (헤어 합성 2경로 + 염색 합성):
- build_image_client: HTTP 타임아웃이 걸린 genai.Client.
  타임아웃이 없으면 멈춘 호출이 Lambda 타임아웃(120초)까지 붙잡혀 재시도할 시간도
  없이 끝난다.
- generate_image: 실패를 분류해 일시적 오류만 재시도한다.
  안전 필터 차단·금지 콘텐츠·4xx 요청 오류는 다시 보내도 같은 결과라 재시도하면
  비용만 최대 3배가 된다.
- inline_image_to_base64: 응답 이미지 → base64 문자열.
"""

import base64
import enum
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from config.settings import settings
from core.logging import logger

# 출력 512 를 지원하지 않고 생각 수준을 고를 수 있는 모델 계열
_NANO_BANANA_PREFIX = "gemini-nano-banana"

# 호출 1회의 HTTP 타임아웃. 정상 호출은 최대 27초(Nano Banana 측정치)라 여유를 두되,
# 한 번 타임아웃이 나도 재시도 1회 + 저장까지 Lambda 120초 안에 끝나도록 잡는다.
GEMINI_IMAGE_TIMEOUT_SECONDS = 45

# 재시도를 포함한 생성 단계 전체의 시간 상한. 다음 시도가 이 시간을 넘길 수 있으면
# (경과 + 대기 + 타임아웃 > 상한) 재시도하지 않는다.
# Lambda 120초에서 업로드 검증·워터마크·저장 몫을 남긴 값이다.
GEMINI_IMAGE_DEADLINE_SECONDS = 100

GEMINI_IMAGE_MAX_ATTEMPTS = 3
GEMINI_IMAGE_RETRY_DELAY = 1.0  # seconds

# 다시 보내도 같은 결과가 나오는 종료 사유 (안전/정책 차단, 저작물 인용)
NON_RETRYABLE_FINISH_REASONS = frozenset(
    {
        "SAFETY",
        "PROHIBITED_CONTENT",
        "BLOCKLIST",
        "SPII",
        "RECITATION",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
    }
)

# 4xx 중 재시도할 가치가 있는 것 (요청 시간 초과, 분당 한도)
_RETRYABLE_CLIENT_CODES = frozenset({408, 429})

# generate_image 실패 사유 (로그용)
FAILURE_BLOCKED = "blocked"  # 프롬프트 차단 / 후보 없음 / 안전 종료 - 재시도 안 함
FAILURE_CLIENT_ERROR = "client_error"  # 4xx (429·408 제외) - 재시도 안 함
FAILURE_NO_IMAGE = "no_image"  # 이미지 없이 정상 종료 - 재시도함
FAILURE_TRANSIENT = "transient_error"  # 5xx·429·타임아웃·연결 오류 - 재시도함
FAILURE_ERROR = "error"  # 그 밖의 예외 (응답 파싱 등) - 재시도 안 함


def is_nano_banana(model: str) -> bool:
    return model.startswith(_NANO_BANANA_PREFIX)


def request_image_size(image_size: Optional[str], model: str) -> Optional[str]:
    """모델에 실제로 요청할 해상도. 512 미지원 모델은 1K 로 받는다."""
    if image_size == "512" and is_nano_banana(model):
        return "1K"
    return image_size


def image_generation_config(
    types, image_size: Optional[str], model: Optional[str] = None
):
    """
    Args:
        types: google.genai.types (호출 측에서 지연 import 한 모듈)
        image_size: 원하는 출력 해상도 ("512" | "1K" | "2K" | None=모델 기본값)
        model: 기본은 settings.GEMINI_IMAGE_MODEL
    """
    model = model or settings.GEMINI_IMAGE_MODEL
    kwargs: Dict[str, Any] = {
        # IMAGE 단독 요청이 TEXT 병행 대비 약 2.6초 빠르고,
        # "텍스트만 반환 → 재시도" 실패 모드를 제거한다.
        "response_modalities": ["IMAGE"],
    }
    size = request_image_size(image_size, model)
    if size:
        kwargs["image_config"] = types.ImageConfig(image_size=size)
    if is_nano_banana(model) and settings.GEMINI_IMAGE_THINKING_LEVEL:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=settings.GEMINI_IMAGE_THINKING_LEVEL
        )
    return types.GenerateContentConfig(**kwargs)


# ========== 클라이언트 ==========


def build_image_client(api_key: Optional[str] = None):
    """이미지 생성용 genai.Client (HTTP 타임아웃 포함).

    HttpOptions.timeout 단위는 밀리초다 (google-genai 가 초로 바꿔 httpx 에 넘긴다).
    SDK 자체 재시도(retry_options)는 켜지 않는다 - 재시도는 generate_image 가 한다.
    """
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key or settings.GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=GEMINI_IMAGE_TIMEOUT_SECONDS * 1000),
    )


# ========== 실패 분류 ==========


def _enum_name(value: Any) -> Optional[str]:
    """SDK enum / 문자열을 대문자 이름으로. 그 밖의 값(None, 목 객체)은 None."""
    if isinstance(value, enum.Enum):
        value = value.value
    if isinstance(value, str) and value:
        return value.upper()
    return None


def is_retryable_error(error: BaseException) -> bool:
    """예외가 다시 시도할 만한 일시적 오류인지.

    - 5xx(ServerError), 429·408 → 재시도
    - 그 밖의 4xx(ClientError: 잘못된 요청·인증·권한·금지 콘텐츠) → 재시도 안 함
    - 타임아웃·연결 오류(httpx, 내장 TimeoutError/ConnectionError) → 재시도
    - 그 밖(응답 파싱 오류 등 코드 문제) → 재시도 안 함 (반복돼도 같은 결과)
    """
    try:
        from google.genai import errors as genai_errors
    except ImportError:  # pragma: no cover - SDK 미설치 환경
        genai_errors = None

    if genai_errors is not None and isinstance(error, genai_errors.APIError):
        code = getattr(error, "code", None)
        if isinstance(code, int):
            return code >= 500 or code in _RETRYABLE_CLIENT_CODES
        # 코드가 없으면 ClientError 만 영구 오류로 본다
        return not isinstance(error, genai_errors.ClientError)

    try:
        import httpx

        if isinstance(error, httpx.TransportError):  # 타임아웃·연결·프로토콜
            return True
    except ImportError:  # pragma: no cover
        pass

    return isinstance(error, (TimeoutError, ConnectionError))


def _blocked_reason(response: Any) -> Optional[str]:
    """재시도해도 소용없는 차단 응답이면 사유 문자열, 아니면 None."""
    feedback = getattr(response, "prompt_feedback", None)
    block_reason = _enum_name(getattr(feedback, "block_reason", None))
    if block_reason:
        return f"prompt_blocked:{block_reason}"

    candidates = getattr(response, "candidates", None)
    if not isinstance(candidates, (list, tuple)) or not candidates:
        return "no_candidates"

    finish_reason = _enum_name(getattr(candidates[0], "finish_reason", None))
    if finish_reason in NON_RETRYABLE_FINISH_REASONS:
        return f"finish_reason:{finish_reason}"
    return None


def _extract_parts(response: Any) -> Tuple[Any, Optional[str]]:
    """(이미지 inline_data, 텍스트) - 첫 번째 후보 기준"""
    content = getattr(response.candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    image = None
    text = None
    for part in parts:
        if getattr(part, "inline_data", None):
            image = part.inline_data
        elif getattr(part, "text", None):
            text = part.text
    return image, text


# ========== 호출 + 재시도 ==========


@dataclass
class ImageGenerationResult:
    """generate_image 결과

    image: 성공 시 응답의 inline_data (data, mime_type), 실패 시 None
    text: 응답에 함께 온 텍스트 (있으면)
    api_calls: 실제 Gemini 호출 수 (재시도 포함, 비용 계측용)
    failure: 실패 사유 (FAILURE_* 상수), 성공이면 None
    detail: 실패 상세 (차단 사유, 예외 문자열 등 - 로그용)
    """

    image: Any = None
    text: Optional[str] = None
    api_calls: int = 0
    failure: Optional[str] = None
    detail: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.image is not None


def generate_image(
    client: Any,
    contents: List[Any],
    config: Any,
    *,
    model: Optional[str] = None,
    max_attempts: int = GEMINI_IMAGE_MAX_ATTEMPTS,
    retry_delay: float = GEMINI_IMAGE_RETRY_DELAY,
    label: str = "이미지 생성",
    sleep: Optional[Callable[[float], None]] = None,
    clock: Optional[Callable[[], float]] = None,
) -> ImageGenerationResult:
    """Gemini 이미지 생성 호출 (일시적 실패만 재시도).

    재시도하지 않는 경우:
    - prompt_feedback.block_reason 이 있음 / 후보가 비어 있음 /
      finish_reason 이 안전·정책 차단 (NON_RETRYABLE_FINISH_REASONS)
    - ClientError 4xx (429·408 제외)
    - 분류되지 않은 예외 (응답 파싱 오류 등)
    - 다음 시도가 GEMINI_IMAGE_DEADLINE_SECONDS 를 넘길 수 있을 때

    재시도하는 경우: 5xx·429·408, 타임아웃·연결 오류, 이미지 없이 정상 종료
    (모델이 가끔 이미지를 빼먹는다 - 기존 동작 유지).

    예외를 밖으로 던지지 않는다. 호출부는 result.success 로 판정한다.
    """
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    model = model or settings.GEMINI_IMAGE_MODEL
    result = ImageGenerationResult()
    started = clock()

    for attempt in range(1, max_attempts + 1):
        retryable = False
        try:
            result.api_calls += 1
            response = client.models.generate_content(
                model=model, contents=contents, config=config
            )

            blocked = _blocked_reason(response)
            if blocked:
                result.failure, result.detail = FAILURE_BLOCKED, blocked
                logger.warning(
                    f"⚠️ {label} 차단 (시도 {attempt}/{max_attempts}, 재시도 안 함): {blocked}"
                )
            else:
                image, text = _extract_parts(response)
                result.text = text
                if image is not None:
                    result.image = image
                    result.failure = result.detail = None
                    logger.info(f"✅ API 호출 성공 (시도 {attempt}/{max_attempts})")
                    return result
                result.failure, result.detail = FAILURE_NO_IMAGE, text
                retryable = True
                logger.warning(
                    f"⚠️ {label} 이미지 미생성 (시도 {attempt}/{max_attempts}), Gemini 응답: {text}"
                )
        except Exception as api_error:
            retryable = is_retryable_error(api_error)
            code = getattr(api_error, "code", None)
            if retryable:
                result.failure = FAILURE_TRANSIENT
            elif isinstance(code, int) and 400 <= code < 500:
                result.failure = FAILURE_CLIENT_ERROR
            else:
                result.failure = FAILURE_ERROR
            result.detail = f"{type(api_error).__name__}: {api_error}"
            logger.warning(
                f"⚠️ {label} API 호출 실패 (시도 {attempt}/{max_attempts}, "
                f"{'재시도' if retryable else '재시도 안 함'}): {result.detail}"
            )

        if not retryable or attempt >= max_attempts:
            break
        elapsed = clock() - started
        if (
            elapsed + retry_delay + GEMINI_IMAGE_TIMEOUT_SECONDS
            > GEMINI_IMAGE_DEADLINE_SECONDS
        ):
            logger.warning(
                f"⚠️ {label} 재시도 생략: 경과 {elapsed:.1f}초 "
                f"(상한 {GEMINI_IMAGE_DEADLINE_SECONDS}초)"
            )
            break
        sleep(retry_delay)

    logger.error(
        f"❌ {label} 실패: {result.failure} ({result.api_calls}회 호출) "
        f"{result.detail or ''}"
    )
    return result


# ========== 응답 이미지 변환 ==========


def inline_image_to_base64(inline_data: Any) -> Tuple[str, str]:
    """응답 inline_data → (base64 문자열, 포맷).

    google-genai(1.56 에서 확인)는 inline_data.data 를 base64 디코드한 원시 bytes 로
    준다. 다만 requirements 는 >=1.52 라 배포 이미지 버전이 다를 수 있고, 이미
    base64 인 값(문자열/bytes)이 와도 깨지지 않도록 예전 판별 로직을 그대로 둔다
    (원시 PNG/JPEG 시그니처면 인코딩, base64 PNG/JPEG 접두사면 그대로).
    """
    data = inline_data.data
    if isinstance(data, str):
        image_base64 = data
    elif isinstance(data, bytes):
        if data[:4] == b"\x89PNG" or data[:3] == b"\xff\xd8\xff":
            image_base64 = base64.b64encode(data).decode("utf-8")
        else:
            try:
                decoded: Optional[str] = data.decode("utf-8")
            except UnicodeDecodeError:
                decoded = None
            if decoded is not None and decoded.startswith(("iVBOR", "/9j/")):
                image_base64 = decoded
            else:
                image_base64 = base64.b64encode(data).decode("utf-8")
    else:
        logger.warning(f"📦 예상치 못한 이미지 데이터 타입: {type(data)}")
        image_base64 = base64.b64encode(bytes(data)).decode("utf-8")

    mime_type = getattr(inline_data, "mime_type", None)
    if isinstance(mime_type, str) and mime_type:
        image_format = mime_type.split("/")[-1]
    else:
        image_format = "png"
    return image_base64, image_format
