"""Pro 계정(미용실용 앱) 손님 기록 API

디자이너가 손님(이름·전화번호·메모)과 상담 기록(날짜·성별·합성 결과 사진·고른 스타일)을
서버에 보관한다. 저장·보관 규칙은 services/pro_records_service.py 참고 (2년 보관).

- 모든 경로는 로그인 필요, Pro 계정만 (손님 계정은 403 pro_only)
- 테이블·버킷이 준비되지 않았으면 503 records_unavailable
- 오류 본문은 {"error": 코드, "message": 한국어 안내} (다른 Pro 거절 응답과 같은 모양)
- 분당 제한은 계정별 (core.rate_limit.pro_account_or_ip_key)
- 이름·전화번호·메모는 로그에 남기지 않는다

GET    /api/pro/customers
POST   /api/pro/customers                                         {"name", "phone"}
PATCH  /api/pro/customers/{customer_id}                           {"name"?, "phone"?, "memo"?}
DELETE /api/pro/customers/{customer_id}
GET    /api/pro/customers/{customer_id}/visits
POST   /api/pro/customers/{customer_id}/visits/{visit_id}/results multipart: style, gender, image
PUT    /api/pro/customers/{customer_id}/visits/{visit_id}/chosen  {"style"}
"""

import json
from typing import Any, Callable, Dict, Optional, Tuple

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.account_type import is_pro
from core.jwt_auth import get_account_type, get_current_user_id
from core.logging import logger, mask_user_id
from core.rate_limit import pro_account_or_ip_key
from database.pro_customer_repository import RecordsTableMissing
from services.pro_records_service import (
    GENDERS,
    MAX_PHOTO_BYTES,
    CustomerNotFound,
    DeleteFailed,
    DuplicatePhone,
    InvalidCustomerInput,
    InvalidStyle,
    RecordsUnavailable,
    SaveFailed,
    TooManyResults,
    VisitNotFound,
    detect_record_photo,
    get_pro_records_service,
    is_valid_id,
    normalize_style,
)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

CUSTOMERS_PATH = "/pro/customers"
CUSTOMER_PATH = CUSTOMERS_PATH + "/{customer_id}"
VISIT_PATH = CUSTOMER_PATH + "/visits/{visit_id}"


# ========== 오류 응답 ==========


def _error(status_code: int, error: str, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": error, "message": message, **extra},
    )


def _pro_only() -> JSONResponse:
    return _error(403, "pro_only", "미용실용 앱 계정에서만 쓸 수 있는 기능입니다.")


def _unavailable() -> JSONResponse:
    return _error(
        503,
        "records_unavailable",
        "손님 기록을 지금 불러올 수 없습니다. 잠시 후 다시 시도해주세요.",
    )


def _invalid_customer() -> JSONResponse:
    return _error(
        400,
        "invalid_customer",
        "이름(1~20자)과 전화번호(예: 010-1234-5678), 메모(500자 이하)를 확인해주세요.",
    )


def _customer_not_found() -> JSONResponse:
    return _error(404, "customer_not_found", "손님 정보를 찾을 수 없습니다.")


def _visit_not_found() -> JSONResponse:
    return _error(404, "visit_not_found", "상담 기록을 찾을 수 없습니다.")


def _invalid_image() -> JSONResponse:
    return _error(
        400, "invalid_image", "3MB 이하의 JPG 또는 PNG 사진만 저장할 수 있습니다."
    )


def _duplicate_phone(customer: Dict[str, Any]) -> JSONResponse:
    return _error(
        409,
        "duplicate_phone",
        "이미 같은 전화번호로 등록된 손님이 있습니다.",
        customer=customer,
    )


async def _read_json(request: Request) -> Optional[Dict[str, Any]]:
    """JSON 객체 본문 (형식이 틀리면 None)"""
    try:
        body = json.loads(await request.body() or b"null")
    except (ValueError, UnicodeDecodeError):
        return None
    return body if isinstance(body, dict) else None


# 서비스 오류 → 응답 (DuplicatePhone 은 기존 손님을 함께 보내므로 따로 처리)
_ERROR_RESPONSES: Tuple[Tuple[Any, Callable[[], JSONResponse]], ...] = (
    ((RecordsUnavailable, RecordsTableMissing), _unavailable),
    (InvalidCustomerInput, _invalid_customer),
    (CustomerNotFound, _customer_not_found),
    (VisitNotFound, _visit_not_found),
    (
        TooManyResults,
        lambda: _error(
            409, "too_many_results", "한 상담에는 결과를 30개까지 저장할 수 있습니다."
        ),
    ),
    (
        InvalidStyle,
        lambda: _error(
            400, "invalid_style", "이 상담에서 합성한 스타일 중에서 골라주세요."
        ),
    ),
    (
        SaveFailed,
        lambda: _error(
            503, "save_failed", "사진을 저장하지 못했습니다. 잠시 후 다시 시도해주세요."
        ),
    ),
    (
        DeleteFailed,
        lambda: _error(
            503, "delete_failed", "삭제하지 못했습니다. 잠시 후 다시 시도해주세요."
        ),
    ),
)


async def _run(
    user_id: str, account_type: str, work: Callable[[], Any]
) -> Tuple[Any, Optional[JSONResponse]]:
    """Pro 확인 → 서비스 호출 (스레드풀). 서비스 오류는 응답으로 바꾼다.

    Returns:
        (결과, None) 또는 (None, 오류 응답)
    """
    if not is_pro(account_type):
        return None, _pro_only()
    try:
        return await run_in_threadpool(work), None
    except DuplicatePhone as e:
        return None, _duplicate_phone(e.customer)
    except Exception as e:
        for error_types, response in _ERROR_RESPONSES:
            if isinstance(e, error_types):
                return None, response()
        logger.error(
            f"❌ 손님 기록 처리 실패: user_id={mask_user_id(user_id)}, "
            f"{type(e).__name__}"
        )
        return None, _unavailable()


# ========== 손님 ==========


@router.get(CUSTOMERS_PATH)
@limiter.limit("60/minute", key_func=pro_account_or_ip_key)
async def list_customers(
    request: Request,
    user_id: str = Depends(get_current_user_id),
    account_type: str = Depends(get_account_type),
):
    """손님 목록 (최근 상담일, 없으면 등록일 기준 최신순)"""
    service = get_pro_records_service()
    customers, error = await _run(
        user_id, account_type, lambda: service.list_customers(user_id)
    )
    return error or {"customers": customers}


@router.post(CUSTOMERS_PATH, status_code=201)
@limiter.limit("30/minute", key_func=pro_account_or_ip_key)
async def create_customer(
    request: Request,
    user_id: str = Depends(get_current_user_id),
    account_type: str = Depends(get_account_type),
):
    """손님 등록. 같은 전화번호의 손님이 있으면 409 duplicate_phone (그 손님 포함)"""
    if not is_pro(account_type):
        return _pro_only()
    body = await _read_json(request)
    if body is None:
        return _invalid_customer()
    service = get_pro_records_service()
    customer, error = await _run(
        user_id,
        account_type,
        lambda: service.create_customer(user_id, body.get("name"), body.get("phone")),
    )
    return error or customer


@router.patch(CUSTOMER_PATH)
@limiter.limit("30/minute", key_func=pro_account_or_ip_key)
async def update_customer(
    request: Request,
    customer_id: str,
    user_id: str = Depends(get_current_user_id),
    account_type: str = Depends(get_account_type),
):
    """이름·전화번호·메모 변경 (보낸 항목만)"""
    if not is_pro(account_type):
        return _pro_only()
    body = await _read_json(request)
    if body is None:
        return _invalid_customer()
    service = get_pro_records_service()
    customer, error = await _run(
        user_id,
        account_type,
        lambda: service.update_customer(user_id, customer_id, body),
    )
    return error or customer


@router.delete(CUSTOMER_PATH)
@limiter.limit("30/minute", key_func=pro_account_or_ip_key)
async def delete_customer(
    request: Request,
    customer_id: str,
    user_id: str = Depends(get_current_user_id),
    account_type: str = Depends(get_account_type),
):
    """손님과 상담 기록·사진 전부 삭제 (사진 먼저). 없는 손님이어도 성공"""
    service = get_pro_records_service()
    _, error = await _run(
        user_id,
        account_type,
        lambda: service.delete_customer(user_id, customer_id),
    )
    return error or {"success": True}


# ========== 상담 기록 ==========


@router.get(CUSTOMER_PATH + "/visits")
@limiter.limit("60/minute", key_func=pro_account_or_ip_key)
async def list_visits(
    request: Request,
    customer_id: str,
    user_id: str = Depends(get_current_user_id),
    account_type: str = Depends(get_account_type),
):
    """손님의 상담 기록 (최신순, 사진 주소는 약 1시간 동안 열린다)"""
    service = get_pro_records_service()
    visits, error = await _run(
        user_id, account_type, lambda: service.list_visits(user_id, customer_id)
    )
    return error or {"visits": visits}


@router.post(VISIT_PATH + "/results", status_code=201)
@limiter.limit("60/minute", key_func=pro_account_or_ip_key)
async def add_visit_result(
    request: Request,
    customer_id: str,
    visit_id: str,
    style: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    user_id: str = Depends(get_current_user_id),
    account_type: str = Depends(get_account_type),
):
    """상담 결과 1건 저장. 상담 id(visit_id)는 앱이 만들고, 첫 결과에서 상담이 생긴다."""
    if not is_pro(account_type):
        return _pro_only()
    service = get_pro_records_service()
    if not service.available:
        return _unavailable()
    if not is_valid_id(visit_id):
        return _error(400, "invalid_visit", "상담 정보가 올바르지 않습니다.")
    clean_style = normalize_style(style)
    if clean_style is None or gender not in GENDERS:
        return _error(400, "invalid_result", "스타일 이름과 성별을 확인해주세요.")
    if image is None:
        return _invalid_image()
    # 3MB 보다 1바이트 더 읽어 크기 초과를 판별한다
    image_bytes = await image.read(MAX_PHOTO_BYTES + 1)
    image_format = detect_record_photo(image_bytes)
    if len(image_bytes) > MAX_PHOTO_BYTES or image_format is None:
        return _invalid_image()

    visit, error = await _run(
        user_id,
        account_type,
        lambda: service.add_result(
            user_id,
            customer_id,
            visit_id,
            clean_style,
            gender,
            image_bytes,
            image_format,
        ),
    )
    return error or {"visit": visit}


@router.put(VISIT_PATH + "/chosen")
@limiter.limit("30/minute", key_func=pro_account_or_ip_key)
async def choose_style(
    request: Request,
    customer_id: str,
    visit_id: str,
    user_id: str = Depends(get_current_user_id),
    account_type: str = Depends(get_account_type),
):
    """손님이 고른 스타일 저장 (그 상담의 결과 스타일 중 하나)"""
    if not is_pro(account_type):
        return _pro_only()
    body = await _read_json(request)
    style = body.get("style") if body else None
    service = get_pro_records_service()
    visit, error = await _run(
        user_id,
        account_type,
        lambda: service.choose_style(user_id, customer_id, visit_id, style),
    )
    return error or {"visit": visit}
