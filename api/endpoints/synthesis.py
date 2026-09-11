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
from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.logging import logger
from core.exceptions import InvalidFileFormatException
from core.jwt_auth import get_optional_user_id
from core.quota import (
    QuotaResult,
    charge_synthesis_quota,
    client_ip_from_request,
)
from core.upload_validation import (
    MAX_ADDITIONAL_INSTRUCTIONS_LENGTH,
    MAX_HAIRSTYLE_NAME_LENGTH,
    sanitize_prompt_text,
    validate_file_extension,
    validate_image_upload,
)
from database.user_repository import get_user_repository
from services.credit_service import get_credit_service
from services.hairstyle_synthesis_service import get_synthesis_service
from services.photo_storage_service import get_photo_storage_service
from services.product_recommendation_service import (
    get_product_recommendation_service,
)
from services.usage_limit_service import get_usage_limit_service

router = APIRouter()

# Rate limiter - synthesis is expensive, so limit more strictly
limiter = Limiter(key_func=get_remote_address)


def _charge_quota(
    user_id: Optional[str],
    device_id: Optional[str],
    client_ip: Optional[str] = None,
    *,
    endpoint: str = "synthesize",
) -> QuotaResult:
    """
    합성 1회분 과금 처리 (실제 로직은 core.quota).

    서비스 팩토리를 호출 시점에 넘겨, 이 모듈을 patch 하는 기존 테스트/호출부가
    그대로 동작하도록 한다.
    """
    return charge_synthesis_quota(
        user_id,
        device_id,
        client_ip,
        endpoint=endpoint,
        credit_service_factory=get_credit_service,
        usage_service_factory=get_usage_limit_service,
    )


def _safe_product_recommendations(
    style_name: Optional[str], gender: Optional[str] = None
) -> list:
    """합성 결과에 얹을 제휴 제품 추천. 실패해도 합성 응답을 깨뜨리면 안 된다."""
    try:
        return get_product_recommendation_service().get_recommendations(
            style_name, gender=gender, limit=3
        )
    except Exception as e:
        logger.warning(f"제품 추천 실패 (무시): {str(e)}")
        return []


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
                "recommended_products": _safe_product_recommendations(
                    hairstyle_name, gender
                ),
            }

        # ===== 3. 과금 (크레딧 또는 레거시 일일 제한) =====
        quota_error, quota, refund = _charge_quota(
            user_id,
            device_id,
            client_ip_from_request(request),
            endpoint="synthesize",
        )
        if quota_error is not None:
            return quota_error

        # ===== 4. 합성 (과금 이후의 모든 실패 경로에서 환불 보장) =====
        try:
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
        except Exception:
            # Gemini 타임아웃/저장 예외 등 - 사용자에게 결과가 없으므로 환불 후 전파
            refund()
            raise

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
            "recommended_products": _safe_product_recommendations(
                hairstyle_name, gender
            ),
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
                "recommended_products": _safe_product_recommendations(None, gender),
            }

        # ===== 3. 과금 (크레딧 또는 레거시 일일 제한) =====
        quota_error, quota, refund = _charge_quota(
            user_id,
            device_id,
            client_ip_from_request(request),
            endpoint="synthesize-with-reference",
        )
        if quota_error is not None:
            return quota_error

        # ===== 4. 합성 (과금 이후의 모든 실패 경로에서 환불 보장) =====
        try:
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
        except Exception:
            # Gemini 타임아웃/저장 예외 등 - 사용자에게 결과가 없으므로 환불 후 전파
            refund()
            raise

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
            "recommended_products": _safe_product_recommendations(None, gender),
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
