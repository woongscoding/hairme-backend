"""Hairstyle synthesis endpoints using Gemini 2.5 Flash Image

과금 정책:
- 로그인 회원 (Authorization: Bearer <JWT>): 크레딧 차감 (합성 실패 시 자동 환불)
- 비로그인 (레거시): device_id 기반 일일 무료 제한 (구버전 앱 호환용, 단계적 폐기 예정)
- 캐시 히트 (같은 사진 + 같은 스타일): 과금 없이 즉시 반환 (Gemini 재호출 방지)

보안:
- 업로드는 확장자 + 매직 바이트 + Pillow 디코딩까지 검증 (core/upload_validation)
- 프롬프트에 삽입되는 사용자 입력은 정제 (프롬프트 인젝션 완화)
- 모든 검증은 과금(크레딧/사용량 차감)보다 먼저 수행
"""

import time
from typing import Any, Callable, Dict, Optional, Tuple

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.logging import logger
from core.exceptions import InvalidFileFormatException
from core.jwt_auth import get_optional_user_id
from core.upload_validation import (
    MAX_ADDITIONAL_INSTRUCTIONS_LENGTH,
    MAX_HAIRSTYLE_NAME_LENGTH,
    sanitize_prompt_text,
    validate_file_extension,
    validate_image_upload,
)
from database.user_repository import get_user_repository
from services.credit_service import InsufficientCreditsError, get_credit_service
from services.hairstyle_synthesis_service import get_synthesis_service
from services.photo_storage_service import get_photo_storage_service
from services.usage_limit_service import get_usage_limit_service
from config.settings import settings

router = APIRouter()

# Rate limiter - synthesis is expensive, so limit more strictly
limiter = Limiter(key_func=get_remote_address)

_NOOP_REFUND: Callable[[], None] = lambda: None


def _charge_quota(
    user_id: Optional[str], device_id: Optional[str]
) -> Tuple[Optional[JSONResponse], Optional[Dict[str, Any]], Callable[[], None]]:
    """
    합성 1회분 과금 처리.

    입력 검증이 모두 끝난 뒤 호출해야 한다
    (유효하지 않은 요청으로 크레딧/사용량이 소진되지 않도록).

    Returns:
        (error_response, quota, refund)
        - error_response: 과금 불가 시 즉시 반환할 응답 (한도 초과/크레딧 부족)
        - quota: 과금 정보 {"mode": "credits", "balance"} 또는 {"mode": "device", ...}
        - refund: 합성 실패 시 호출할 환불 함수 (크레딧 모드만 실제 환불)
    """
    # ===== 회원: 크레딧 차감 =====
    if user_id:
        cost = settings.SYNTHESIS_CREDIT_COST
        try:
            balance = get_credit_service().consume(user_id, cost, reason="synthesis")
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
                _NOOP_REFUND,
            )
        except ValueError:
            raise HTTPException(status_code=401, detail="유효하지 않은 사용자입니다.")
        except Exception as e:
            logger.error(f"크레딧 차감 실패 (blocking): {str(e)}")
            raise HTTPException(
                status_code=503,
                detail="크레딧 확인 서비스에 일시적 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            )

        def refund() -> None:
            try:
                get_credit_service().grant(user_id, cost, reason="refund")
                logger.info(f"합성 실패 크레딧 환불: user_id={user_id}, +{cost}")
            except Exception as refund_error:
                logger.error(f"❌ 크레딧 환불 실패: {str(refund_error)}")

        return None, {"mode": "credits", "balance": balance}, refund

    # ===== 비로그인: 레거시 device_id 일일 제한 =====
    trimmed_device_id = (device_id or "").strip()
    if not trimmed_device_id:
        raise HTTPException(
            status_code=401,
            detail="로그인이 필요합니다. (구버전 앱은 device_id 필수)",
        )

    try:
        usage_result = get_usage_limit_service().check_and_increment_usage(
            trimmed_device_id
        )
    except Exception as e:
        logger.error(f"Usage limit check failed (blocking): {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="사용량 확인 서비스에 일시적 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        )

    if not usage_result["allowed"]:
        daily_limit = settings.DAILY_SYNTHESIS_LIMIT
        return (
            JSONResponse(
                status_code=429,
                content={
                    "error": "daily_limit_exceeded",
                    "message": f"오늘의 무료 합성 횟수({daily_limit}회)를 모두 사용했습니다.",
                    "daily_limit": daily_limit,
                    "used": usage_result["used"],
                    "remaining": 0,
                },
            ),
            None,
            _NOOP_REFUND,
        )

    quota = {
        "mode": "device",
        "daily_limit": usage_result["daily_limit"],
        "used": usage_result["used"],
        "remaining": usage_result["remaining"],
    }
    # 레거시 흐름은 기존 동작 유지 (합성 실패해도 차감 롤백 없음)
    return None, quota, _NOOP_REFUND


def _store_result(
    user_id: Optional[str],
    image_data: bytes,
    cache_key: str,
    image_base64: str,
    image_format: str,
    hairstyle_name: Optional[str],
) -> Optional[str]:
    """합성 성공 후 저장 처리 (캐시 + 회원 결과 + 동의 시 원본). 실패해도 응답에 영향 없음"""
    storage = get_photo_storage_service()
    if not storage.enabled:
        return None

    storage.save_cached_result(cache_key, image_base64, image_format)

    result_url = None
    if user_id:
        result_url = storage.save_user_result(
            user_id, image_base64, image_format, hairstyle_name
        )
        # AI 학습 활용에 동의한 회원만 원본 저장
        try:
            user = get_user_repository().get_by_id(user_id)
            if user and user.get("training_consent"):
                storage.save_original_photo(user_id, image_data)
        except Exception as e:
            logger.warning(f"원본 저장 확인 실패 (무시): {str(e)}")

    return result_url


@router.post("/synthesize")
@limiter.limit("5/minute")  # 분당 5회 제한 (이미지 생성은 비용이 높음)
async def synthesize_hairstyle(
    request: Request,
    file: UploadFile = File(..., description="사용자 얼굴 사진"),
    hairstyle_name: str = Form(
        ..., description="적용할 헤어스타일 이름 (예: 투블럭컷)"
    ),
    gender: str = Form("male", description="성별 (male/female)"),
    device_id: Optional[str] = Form(
        None, description="디바이스 고유 ID (비로그인 레거시 흐름용)"
    ),
    additional_instructions: Optional[str] = Form(
        None, description="추가 스타일링 요청 (선택)"
    ),
    user_id: Optional[str] = Depends(get_optional_user_id),
):
    """
    헤어스타일 합성 API

    사용자 얼굴 사진에 선택한 헤어스타일을 적용한 이미지를 생성합니다.

    - 로그인(Bearer 토큰) 시: 크레딧 1 차감, 결과가 마이페이지에 저장됨
    - 비로그인 시: device_id 기반 일일 무료 제한 (레거시)
    - 같은 사진+스타일 재요청 시: 캐시 반환 (과금 없음)

    Returns:
        {
            "success": true,
            "image_base64": "...",
            "image_format": "png",
            "message": "...",
            "processing_time": 3.5,
            "cached": false,
            "result_url": "https://...(회원만)",
            "quota": {"mode": "credits", "balance": 4}
        }
    """
    start_time = time.time()

    try:
        # ===== 1. 입력 검증 (과금 전에 수행) =====
        validate_file_extension(file.filename)

        if gender not in ["male", "female"]:
            raise HTTPException(
                status_code=400, detail="gender는 'male' 또는 'female'만 가능합니다."
            )

        # Gemini 프롬프트에 삽입되는 사용자 입력 정제 (프롬프트 인젝션 완화)
        # 캐시 키 계산 전에 정제해야 같은 요청이 같은 키를 갖는다
        hairstyle_name = sanitize_prompt_text(
            hairstyle_name, MAX_HAIRSTYLE_NAME_LENGTH, "헤어스타일 이름"
        )
        if not hairstyle_name:
            raise HTTPException(
                status_code=400, detail="헤어스타일 이름이 올바르지 않습니다."
            )
        if additional_instructions:
            additional_instructions = sanitize_prompt_text(
                additional_instructions,
                MAX_ADDITIONAL_INSTRUCTIONS_LENGTH,
                "추가 요청",
            )

        # 실제 크기 + 매직 바이트 + Pillow 디코딩 검증
        image_data = await file.read()
        validate_image_upload(image_data)

        logger.info(f"🎨 합성 요청: {hairstyle_name} ({gender}), file={file.filename}")

        # ===== 2. 캐시 확인 (과금 전에 - 히트 시 무료) =====
        storage = get_photo_storage_service()
        cache_key = storage.build_cache_key(
            image_data,
            hairstyle_name.encode("utf-8"),
            gender.encode("utf-8"),
            (additional_instructions or "").encode("utf-8"),
        )
        cached = storage.get_cached_result(cache_key)
        if cached:
            return {
                "success": True,
                "image_base64": cached["image_base64"],
                "image_format": cached["image_format"],
                "message": f"'{hairstyle_name}' 스타일이 적용되었습니다.",
                "processing_time": round(time.time() - start_time, 2),
                "cached": True,
                "result_url": None,
                "quota": None,
            }

        # ===== 3. 과금 (크레딧 또는 레거시 일일 제한) =====
        quota_error, quota, refund = _charge_quota(user_id, device_id)
        if quota_error is not None:
            return quota_error

        # ===== 4. 합성 =====
        service = get_synthesis_service()
        result = service.synthesize_hairstyle(
            image_data=image_data,
            hairstyle_name=hairstyle_name,
            gender=gender,
            additional_instructions=additional_instructions,
        )

        processing_time = round(time.time() - start_time, 2)

        if not result["success"]:
            refund()
            logger.warning(f"⚠️ 합성 실패: {result['message']}")
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": result["message"],
                    "processing_time": processing_time,
                },
            )

        # ===== 5. 저장 (캐시 + 회원 결과 + 동의 시 원본) =====
        result_url = _store_result(
            user_id,
            image_data,
            cache_key,
            result["image_base64"],
            result["image_format"],
            hairstyle_name,
        )

        logger.info(f"✅ 합성 완료: {hairstyle_name} ({processing_time}초)")
        return {
            "success": True,
            "image_base64": result["image_base64"],
            "image_format": result["image_format"],
            "message": result["message"],
            "processing_time": processing_time,
            "cached": False,
            "result_url": result_url,
            "quota": quota,
        }

    except InvalidFileFormatException as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "invalid_file_format",
                "message": str(e),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 합성 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )


@router.post("/synthesize-with-reference")
@limiter.limit("3/minute")  # 레퍼런스 기반은 더 제한적
async def synthesize_with_reference(
    request: Request,
    user_photo: UploadFile = File(..., description="사용자 얼굴 사진"),
    reference_photo: UploadFile = File(..., description="참고할 헤어스타일 사진"),
    gender: str = Form("male", description="성별 (male/female)"),
    device_id: Optional[str] = Form(
        None, description="디바이스 고유 ID (비로그인 레거시 흐름용)"
    ),
    user_id: Optional[str] = Depends(get_optional_user_id),
):
    """
    레퍼런스 이미지 기반 헤어스타일 합성 API

    참고 이미지의 헤어스타일을 사용자 얼굴에 적용합니다.
    과금 정책은 /synthesize와 동일 (회원: 크레딧, 비로그인: device_id 일일 제한).
    """
    start_time = time.time()

    try:
        # ===== 1. 입력 검증 (과금 전에 수행) =====
        validate_file_extension(user_photo.filename)
        validate_file_extension(reference_photo.filename)

        if gender not in ["male", "female"]:
            raise HTTPException(
                status_code=400, detail="gender는 'male' 또는 'female'만 가능합니다."
            )

        # 실제 크기 + 매직 바이트 + Pillow 디코딩 검증
        user_image_data = await user_photo.read()
        validate_image_upload(user_image_data)

        reference_image_data = await reference_photo.read()
        validate_image_upload(reference_image_data)

        logger.info(f"🎨 레퍼런스 합성 요청: {gender}")

        # ===== 2. 캐시 확인 (과금 전에 - 히트 시 무료) =====
        storage = get_photo_storage_service()
        cache_key = storage.build_cache_key(
            user_image_data,
            reference_image_data,
            gender.encode("utf-8"),
            b"reference",
        )
        cached = storage.get_cached_result(cache_key)
        if cached:
            return {
                "success": True,
                "image_base64": cached["image_base64"],
                "image_format": cached["image_format"],
                "message": "레퍼런스 스타일이 적용되었습니다.",
                "processing_time": round(time.time() - start_time, 2),
                "cached": True,
                "result_url": None,
                "quota": None,
            }

        # ===== 3. 과금 (크레딧 또는 레거시 일일 제한) =====
        quota_error, quota, refund = _charge_quota(user_id, device_id)
        if quota_error is not None:
            return quota_error

        # ===== 4. 합성 =====
        service = get_synthesis_service()
        result = service.synthesize_with_reference(
            user_image_data=user_image_data,
            reference_image_data=reference_image_data,
            gender=gender,
        )

        processing_time = round(time.time() - start_time, 2)

        if not result["success"]:
            refund()
            logger.warning(f"⚠️ 레퍼런스 합성 실패: {result['message']}")
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": result["message"],
                    "processing_time": processing_time,
                },
            )

        # ===== 5. 저장 (캐시 + 회원 결과 + 동의 시 원본) =====
        result_url = _store_result(
            user_id,
            user_image_data,
            cache_key,
            result["image_base64"],
            result["image_format"],
            hairstyle_name=None,
        )

        logger.info(f"✅ 레퍼런스 합성 완료 ({processing_time}초)")
        return {
            "success": True,
            "image_base64": result["image_base64"],
            "image_format": result["image_format"],
            "message": result["message"],
            "processing_time": processing_time,
            "cached": False,
            "result_url": result_url,
            "quota": quota,
        }

    except InvalidFileFormatException as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "invalid_file_format",
                "message": str(e),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 레퍼런스 합성 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )
