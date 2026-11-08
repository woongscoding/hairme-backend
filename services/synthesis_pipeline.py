"""합성 계열 엔드포인트가 공유하는 요청 처리 단계

대상: api/endpoints/synthesis.py (POST /api/v2/synthesize)
      api/endpoints/hair_color.py (POST /api/hair-color/synthesize)

두 엔드포인트의 순서는 같다:
  입력 검증 → [캐시: 스타일 합성만] → 일 예산 상한(비로그인만, 503)
  → 중복 요청 잠금(409) → 과금 → Gemini 호출 → record_api_calls
  → 실패 시 환불(422) → 응답 → finally 에서 잠금 해제

여기에는 두 엔드포인트가 "똑같이" 하는 부분만 둔다.
입력 필드·캐시·암묵 신호·저장·성공 응답 스키마처럼 서로 다른 부분은 각 엔드포인트에 남긴다.

환불 규칙 (돈이 걸린 경로):
- 과금(charge_synthesis_quota) 전에 끝나는 경로(검증 400, 예산 503, 잠금 409,
  한도 402/429)에는 환불이 없다. 이 모듈의 사전 거절 단계는 환불 함수를 받지 않는다.
- 과금 뒤의 실패 경로는 refund() 를 부른다. 한 요청에서 여러 경로가 겹쳐 refund() 가
  두 번 불릴 수 있지만(예: 422 처리 중 예외 → refund_on_error), core.quota 의 환불
  함수는 _refund_once 로 감싸져 있어 실제 환불은 정확히 1회다.

테스트 patch 경로:
- 예산/잠금/집계/로그 함수는 SynthesisPipeline 생성 시점에 엔드포인트 모듈이 넘긴다.
  그래서 기존처럼 api.endpoints.synthesis.daily_budget_exceeded 등을 patch 하면
  그대로 적용된다 (synthesis._charge_quota 가 서비스 팩토리를 넘기는 것과 같은 이유).
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from core.logging import log_structured
from core.quota import refunded_balance
from core.synthesis_budget import (
    budget_exceeded_response_body,
    daily_budget_exceeded,
    record_api_calls,
)
from core.synthesis_lock import (
    NOOP_RELEASE,
    acquire_synthesis_lock,
    duplicate_request_response_body,
    lock_subject,
)

INTERNAL_ERROR_DETAIL = "서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

PreChargeResult = Tuple[Optional[JSONResponse], Callable[[], None]]


def charge_mode(user_id: Optional[str], quota: Optional[dict] = None) -> str:
    """구조화 로그의 mode 값 (credits=회원 과금, device=비로그인).

    차감 결과가 있으면 그 mode 를 쓴다. 캐시 히트는 과금 경로를 타지 않으므로
    호출부에서 "cache" 를 직접 넣는다.
    """
    if quota and quota.get("mode"):
        return str(quota["mode"])
    return "credits" if user_id else "device"


def invalid_file_format_response(error: Exception) -> JSONResponse:
    """업로드 형식 오류 400 본문 (두 엔드포인트 공통)"""
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": "invalid_file_format",
            "message": str(error),
        },
    )


def internal_error() -> HTTPException:
    """예상하지 못한 예외의 500 (내부 메시지는 응답에 노출하지 않는다)"""
    return HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL)


@contextmanager
def refund_on_error(refund: Callable[[], None]) -> Iterator[None]:
    """블록 안에서 예외가 나면 환불한 뒤 그대로 다시 던진다.

    과금 직후부터 사용자에게 결과를 넘기기 전까지(Gemini 호출, 워터마크, 저장)를 감싼다.
    블록 안에서 이미 환불한 경로(422)가 뒤이어 예외를 내도, 환불 함수가
    _refund_once 로 감싸져 있어 두 번 환불되지 않는다.
    """
    try:
        yield
    except Exception:
        refund()
        raise


def synthesis_rejected_body(
    message: str,
    processing_time: float,
    quota: Optional[Dict[str, Any]],
    **extra: Any,
) -> dict:
    """합성 실패 422 본문. extra(염색의 color_name/color_hex 등)는 message 뒤에 들어간다."""
    return {
        "success": False,
        "error": "synthesis_rejected",
        "message": message,
        **extra,
        "processing_time": processing_time,
        # 차감분은 되돌렸다는 것을 앱이 그대로 보여줄 수 있게
        "refunded": True,
        "balance": refunded_balance(quota),
    }


@dataclass
class SynthesisPipeline:
    """합성 요청 1건의 공통 단계 (예산·잠금·호출 집계·실패 로그/응답).

    Args:
        endpoint: 로그/집계/잠금에 남기는 엔드포인트 이름 (예: "synthesize")
        failed_event: 실패 로그 이벤트 이름 (예: "synthesis_failed")
        user_id: 로그인 회원 ID. None 이면 비로그인
        start_time: 요청 시작 시각 (processing_time 계산용)
        log / budget_exceeded / acquire_lock / record_calls:
            엔드포인트 모듈이 호출 시점에 자기 이름을 넘긴다 (모듈 docstring 의 patch 경로 참고)
    """

    endpoint: str
    failed_event: str
    user_id: Optional[str]
    start_time: float
    log: Callable[[str, dict], None] = log_structured
    budget_exceeded: Callable[..., bool] = daily_budget_exceeded
    acquire_lock: Callable[..., Tuple[bool, Callable[[], None]]] = (
        acquire_synthesis_lock
    )
    record_calls: Callable[..., None] = record_api_calls

    @property
    def authenticated(self) -> bool:
        return self.user_id is not None

    def elapsed(self) -> float:
        return round(time.time() - self.start_time, 2)

    def mode(self, quota: Optional[dict] = None) -> str:
        return charge_mode(self.user_id, quota)

    def log_rejected(
        self,
        reason: str,
        status_code: int,
        fields: Optional[dict] = None,
    ) -> None:
        """Gemini 를 부르기 전 거절(예산 503 / 잠금 409 / 한도 402·429) 로그.

        api_calls 필드는 넣지 않는다 - 로그 합산 때 실제 호출로 오인되지 않도록.
        """
        self.log(
            self.failed_event,
            {
                "endpoint": self.endpoint,
                "mode": self.mode(),
                **(fields or {}),
                "reason": reason,
                "status_code": status_code,
                "processing_time": self.elapsed(),
                "authenticated": self.authenticated,
            },
        )

    def pre_charge_guard(
        self,
        device_id: Optional[str],
        request_key: str,
        *,
        log_fields: Optional[dict] = None,
    ) -> PreChargeResult:
        """일 예산 상한 + 중복 요청 잠금. 반드시 과금 전에 부른다.

        Returns:
            (rejection, release_lock)
            - rejection 이 있으면 그대로 반환한다 (503 예산 / 409 중복). release 는 no-op
            - 없으면 진행. release_lock() 은 호출부가 finally 에서 부른다
        """
        # 예산은 잠금보다 먼저 본다 (거절될 요청이 잠금을 잡으면 안 된다). 회원은 전원 통과
        if self.user_id is None and self.budget_exceeded(endpoint=self.endpoint):
            self.log_rejected("daily_budget_exceeded", 503, log_fields)
            return (
                JSONResponse(status_code=503, content=budget_exceeded_response_body()),
                NOOP_RELEASE,
            )

        acquired, release_lock = self.acquire_lock(
            lock_subject(self.user_id, device_id), request_key, endpoint=self.endpoint
        )
        if not acquired:
            self.log_rejected("duplicate_in_flight", 409, log_fields)
            return (
                JSONResponse(
                    status_code=409, content=duplicate_request_response_body()
                ),
                NOOP_RELEASE,
            )
        return None, release_lock

    def record_gemini_calls(self, result: dict) -> int:
        """서비스 결과의 실제 Gemini 호출 수를 일 예산 집계에 더하고 그 값을 돌려준다.

        성공/실패 모두 비용은 이미 나갔으므로 둘 다 기록한다.
        """
        api_calls = result.get("api_calls", 0)
        self.record_calls(api_calls, endpoint=self.endpoint)
        return api_calls

    def reject_synthesis(
        self,
        result: dict,
        quota: Optional[Dict[str, Any]],
        refund: Callable[[], None],
        api_calls: int,
        processing_time: float,
        *,
        log_fields: Optional[dict] = None,
        body_extra: Optional[dict] = None,
    ) -> JSONResponse:
        """Gemini 가 결과를 못 만든 경우: 환불 → 실패 로그 → 422."""
        refund()
        self.log(
            self.failed_event,
            {
                "endpoint": self.endpoint,
                "mode": self.mode(quota),
                **(log_fields or {}),
                "reason": "synthesis_rejected",
                "status_code": 422,
                "api_calls": api_calls,
                "processing_time": processing_time,
                "refunded": True,
                "authenticated": self.authenticated,
            },
        )
        return JSONResponse(
            status_code=422,
            content=synthesis_rejected_body(
                result["message"], processing_time, quota, **(body_extra or {})
            ),
        )
