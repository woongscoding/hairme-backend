"""합성 1회분 과금(크레딧 / 무료 한도) 공통 로직

합성 계열 엔드포인트(hairstyle synthesis, hair color synthesis)가 동일한
과금 정책을 쓰도록 한 곳에 모았다.

정책:
- 로그인 회원 (Authorization: Bearer <JWT>): 크레딧 차감 (합성 실패 시 자동 환불)
- 비로그인 (레거시): device_id 기반 일일 무료 제한
  + 클라이언트 IP 기준 일일 상한 (device_id 로테이션으로 무제한 사용하는 것 차단)

주의:
- 반드시 모든 입력 검증이 끝난 뒤에 호출할 것 (잘못된 요청으로 과금되지 않도록)
- device_id는 사용량 테이블의 파티션 키를 공유하므로 반드시 검증된 값만 사용할 것
  (services.usage_limit_service.validate_device_id)
"""

import hashlib
import ipaddress
from typing import Any, Callable, Dict, Optional, Tuple

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from config.settings import settings
from core.logging import log_structured, logger
from services.credit_service import InsufficientCreditsError, get_credit_service
from services.usage_limit_service import get_usage_limit_service, validate_device_id

NOOP_REFUND: Callable[[], None] = lambda: None

QuotaResult = Tuple[
    Optional[JSONResponse], Optional[Dict[str, Any]], Callable[[], None]
]


def mask_device_id(device_id: Optional[str]) -> str:
    """device_id 마스킹: 앞 4자 + sha256 앞 8자

    로그에 원본 식별자를 남기지 않으면서도 같은 기기를 셀 수 있게 한다.
    """
    if not device_id:
        return "unknown"
    digest = hashlib.sha256(device_id.encode("utf-8")).hexdigest()[:8]
    return f"{device_id[:4]}~{digest}"


def mask_ip(client_ip: Optional[str]) -> str:
    """IP 마스킹: IPv4는 마지막 옥텟을 0으로, IPv6는 /48 로 절단"""
    if not client_ip:
        return "unknown"
    try:
        address = ipaddress.ip_address(client_ip)
    except ValueError:
        return "invalid"
    if address.version == 4:
        octets = client_ip.split(".")
        return ".".join(octets[:3] + ["0"])
    return str(ipaddress.ip_network(f"{address}/48", strict=False).network_address)


def client_ip_from_request(request: Optional[Request]) -> Optional[str]:
    """요청의 클라이언트 IP (Mangum이 Lambda sourceIp로 채운다). 없으면 None"""
    if request is None or request.client is None:
        return None
    return request.client.host or None


def _ip_limit_response(limit: int) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": "daily_limit_exceeded",
            "message": f"일일 무료 합성 한도를 초과했습니다. ({limit}회) 로그인 후 이용해주세요.",
            "limit_type": "ip",
            "daily_limit": limit,
            "remaining": 0,
        },
    )


def _device_limit_response(daily_limit: int, used: int) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": "daily_limit_exceeded",
            "message": f"오늘의 무료 합성 횟수({daily_limit}회)를 모두 사용했습니다.",
            "limit_type": "device",
            "daily_limit": daily_limit,
            "used": used,
            "remaining": 0,
        },
    )


def charge_synthesis_quota(
    user_id: Optional[str],
    device_id: Optional[str],
    client_ip: Optional[str] = None,
    *,
    endpoint: str = "unknown",
    credit_service_factory: Optional[Callable[[], Any]] = None,
    usage_service_factory: Optional[Callable[[], Any]] = None,
) -> QuotaResult:
    """
    합성 1회분 과금 처리.

    입력 검증이 모두 끝난 뒤 호출해야 한다
    (유효하지 않은 요청으로 크레딧/사용량이 소진되지 않도록).

    Args:
        user_id: 로그인 사용자 ID (JWT). None이면 비로그인 흐름
        device_id: 비로그인 흐름의 디바이스 ID
        client_ip: 비로그인 IP 일일 상한용 (None이면 IP 상한 생략)
        endpoint: 레거시 흐름 계측 로그에 남길 호출 엔드포인트 이름
        credit_service_factory / usage_service_factory:
            테스트/호출부에서 서비스 팩토리를 주입하기 위한 훅 (기본은 모듈 전역)

    Returns:
        (error_response, quota, refund)
        - error_response: 과금 불가 시 즉시 반환할 응답 (한도 초과/크레딧 부족)
        - quota: 과금 정보 {"mode": "credits", "balance"} 또는 {"mode": "device", ...}
        - refund: 합성 실패 시 호출할 환불 함수
    """
    credit_factory = credit_service_factory or get_credit_service
    usage_factory = usage_service_factory or get_usage_limit_service

    # ===== 회원: 크레딧 차감 =====
    if user_id:
        cost = settings.SYNTHESIS_CREDIT_COST
        try:
            balance = credit_factory().consume(user_id, cost, reason="synthesis")
        except InsufficientCreditsError as e:
            return (
                JSONResponse(
                    status_code=402,
                    content={
                        "error": "insufficient_credits",
                        "message": "크레딧이 부족합니다. 크레딧을 충전해주세요.",
                        "balance": e.balance,
                    },
                ),
                None,
                NOOP_REFUND,
            )
        except ValueError:
            raise HTTPException(status_code=401, detail="유효하지 않은 사용자입니다.")
        except Exception as e:
            logger.error(f"크레딧 차감 실패 (blocking): {str(e)}")
            raise HTTPException(
                status_code=503,
                detail="크레딧 확인 서비스에 일시적 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            )

        def refund_credits() -> None:
            try:
                credit_factory().grant(user_id, cost, reason="refund")
                logger.info(f"합성 실패 크레딧 환불: user_id={user_id}, +{cost}")
            except Exception as refund_error:
                logger.error(f"❌ 크레딧 환불 실패: {str(refund_error)}")

        return None, {"mode": "credits", "balance": balance}, refund_credits

    # ===== 비로그인: 레거시 device_id 일일 제한 =====
    # 킬 스위치: 레거시 흐름을 닫으면 카운터를 건드리기 전에 401 로 끊는다.
    if not settings.LEGACY_DEVICE_FLOW_ENABLED:
        raise HTTPException(
            status_code=401,
            detail="로그인이 필요합니다. 앱을 최신 버전으로 업데이트한 뒤 로그인해주세요.",
        )

    trimmed_device_id = (device_id or "").strip()
    if not trimmed_device_id:
        raise HTTPException(
            status_code=401,
            detail="로그인이 필요합니다. (구버전 앱은 device_id 필수)",
        )

    try:
        trimmed_device_id = validate_device_id(trimmed_device_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 레거시(비로그인) 과금 1건당 구조화 로그 1줄.
    # CloudWatch Logs Insights 로 플래그를 내리기 전 잔존 사용량을 측정한다.
    # 원본 device_id / IP 는 절대 남기지 않는다 (마스킹 필수).
    log_structured(
        "legacy_device_flow",
        {
            "endpoint": endpoint,
            "device_id": mask_device_id(trimmed_device_id),
            "ip": mask_ip(client_ip),
        },
    )

    usage_service = usage_factory()

    # (1) IP 일일 상한 - device_id를 매번 새로 만들어도 우회할 수 없도록
    ip_key: Optional[str] = None
    if client_ip:
        ip_limit = settings.ANON_IP_DAILY_SYNTHESIS_LIMIT
        ip_key = f"ip#{client_ip}"
        try:
            ip_allowed = usage_service.increment_daily_counter(ip_key, ip_limit)
        except Exception as e:
            logger.error(f"IP 일일 한도 확인 실패 (blocking): {str(e)}")
            raise HTTPException(
                status_code=503,
                detail="사용량 확인 서비스에 일시적 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            )
        if not ip_allowed:
            logger.warning(f"IP 일일 합성 한도 초과: ip={client_ip}")
            return _ip_limit_response(ip_limit), None, NOOP_REFUND

    # (2) device_id 일일 제한
    try:
        usage_result = usage_service.check_and_increment_usage(trimmed_device_id)
    except Exception as e:
        if ip_key:
            usage_service.decrement_daily_counter(ip_key)
        logger.error(f"Usage limit check failed (blocking): {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="사용량 확인 서비스에 일시적 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        )

    if not usage_result["allowed"]:
        if ip_key:
            usage_service.decrement_daily_counter(ip_key)
        return (
            _device_limit_response(
                settings.DAILY_SYNTHESIS_LIMIT, usage_result["used"]
            ),
            None,
            NOOP_REFUND,
        )

    quota = {
        "mode": "device",
        "daily_limit": usage_result["daily_limit"],
        "used": usage_result["used"],
        "remaining": usage_result["remaining"],
    }

    def refund_free_quota() -> None:
        """합성 실패 시 무료 한도 복구 (best effort)"""
        usage_service.decrement_usage(trimmed_device_id)
        if ip_key:
            usage_service.decrement_daily_counter(ip_key)

    return None, quota, refund_free_quota
