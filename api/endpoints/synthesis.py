"""Hairstyle synthesis endpoints using Gemini 2.5 Flash Image"""

import time
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.logging import logger
from core.exceptions import InvalidFileFormatException
from core.upload_validation import (
    MAX_ADDITIONAL_INSTRUCTIONS_LENGTH,
    MAX_HAIRSTYLE_NAME_LENGTH,
    sanitize_prompt_text,
    validate_file_extension,
    validate_image_upload,
)
from services.hairstyle_synthesis_service import get_synthesis_service
from services.usage_limit_service import get_usage_limit_service
from config.settings import settings

router = APIRouter()

# Rate limiter - synthesis is expensive, so limit more strictly
limiter = Limiter(key_func=get_remote_address)


@router.post("/synthesize")
@limiter.limit("5/minute")  # 분당 5회 제한 (이미지 생성은 비용이 높음)
async def synthesize_hairstyle(
    request: Request,
    file: UploadFile = File(..., description="사용자 얼굴 사진"),
    hairstyle_name: str = Form(
        ..., description="적용할 헤어스타일 이름 (예: 투블럭컷)"
    ),
    gender: str = Form("male", description="성별 (male/female)"),
    device_id: str = Form(..., description="디바이스 고유 ID (일일 사용량 제한용)"),
    additional_instructions: Optional[str] = Form(
        None, description="추가 스타일링 요청 (선택)"
    ),
):
    """
    헤어스타일 합성 API

    사용자 얼굴 사진에 선택한 헤어스타일을 적용한 이미지를 생성합니다.

    Args:
        file: 사용자 얼굴 사진 (JPG, PNG, WEBP)
        hairstyle_name: 적용할 헤어스타일 이름
        gender: 성별 (male/female)
        device_id: 디바이스 고유 ID (일일 사용량 제한용)
        additional_instructions: 추가 스타일링 요청 (선택)

    Returns:
        {
            "success": true,
            "image_base64": "base64 encoded image",
            "image_format": "png",
            "message": "헤어스타일이 적용되었습니다.",
            "processing_time": 3.5
        }
    """
    start_time = time.time()

    try:
        # ========== 1. 입력 검증 (사용량 차감 전에 수행) ==========
        trimmed_device_id = device_id.strip()
        if not trimmed_device_id:
            raise HTTPException(status_code=400, detail="device_id는 필수입니다.")

        # File validation
        validate_file_extension(file.filename)

        # Gender validation
        if gender not in ["male", "female"]:
            raise HTTPException(
                status_code=400, detail="gender는 'male' 또는 'female'만 가능합니다."
            )

        # Gemini 프롬프트에 삽입되는 사용자 입력 정제 (프롬프트 인젝션 완화)
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

        # Read image data + 실제 크기/내용 검증
        image_data = await file.read()
        validate_image_upload(image_data)

        # ========== 2. Daily usage limit: atomic check + increment ==========
        # 유효하지 않은 요청으로 타인의 사용량이 소진되지 않도록 검증 후 차감
        try:
            usage_service = get_usage_limit_service()
            usage_result = usage_service.check_and_increment_usage(trimmed_device_id)

            if not usage_result["allowed"]:
                daily_limit = settings.DAILY_SYNTHESIS_LIMIT
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "daily_limit_exceeded",
                        "message": f"오늘의 무료 합성 횟수({daily_limit}회)를 모두 사용했습니다.",
                        "daily_limit": daily_limit,
                        "used": usage_result["used"],
                        "remaining": 0,
                    },
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Usage limit check failed (blocking): {str(e)}")
            raise HTTPException(
                status_code=503,
                detail="사용량 확인 서비스에 일시적 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            )

        logger.info(f"🎨 합성 요청: {hairstyle_name} ({gender}), file={file.filename}")

        # Get synthesis service and process
        service = get_synthesis_service()
        result = service.synthesize_hairstyle(
            image_data=image_data,
            hairstyle_name=hairstyle_name,
            gender=gender,
            additional_instructions=additional_instructions,
        )

        processing_time = round(time.time() - start_time, 2)

        if result["success"]:
            logger.info(f"✅ 합성 완료: {hairstyle_name} ({processing_time}초)")
            return {
                "success": True,
                "image_base64": result["image_base64"],
                "image_format": result["image_format"],
                "message": result["message"],
                "processing_time": processing_time,
            }
        else:
            logger.warning(f"⚠️ 합성 실패: {result['message']}")
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": result["message"],
                    "processing_time": processing_time,
                },
            )

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
    device_id: str = Form(..., description="디바이스 고유 ID (일일 사용량 제한용)"),
):
    """
    레퍼런스 이미지 기반 헤어스타일 합성 API

    참고 이미지의 헤어스타일을 사용자 얼굴에 적용합니다.

    Args:
        user_photo: 사용자 얼굴 사진
        reference_photo: 참고할 헤어스타일 사진
        gender: 성별 (male/female)

    Returns:
        {
            "success": true,
            "image_base64": "base64 encoded image",
            "image_format": "png",
            "message": "레퍼런스 스타일이 적용되었습니다.",
            "processing_time": 4.2
        }
    """
    start_time = time.time()

    try:
        # ========== 1. 입력 검증 (사용량 차감 전에 수행) ==========
        trimmed_device_id = device_id.strip()
        if not trimmed_device_id:
            raise HTTPException(status_code=400, detail="device_id는 필수입니다.")

        # Validate user photo
        validate_file_extension(user_photo.filename)

        # Validate reference photo
        validate_file_extension(reference_photo.filename)

        # Gender validation
        if gender not in ["male", "female"]:
            raise HTTPException(
                status_code=400, detail="gender는 'male' 또는 'female'만 가능합니다."
            )

        # Read image data + 실제 크기/내용 검증
        user_image_data = await user_photo.read()
        validate_image_upload(user_image_data)

        reference_image_data = await reference_photo.read()
        validate_image_upload(reference_image_data)

        # ========== 2. Daily usage limit: atomic check + increment ==========
        # 유효하지 않은 요청으로 타인의 사용량이 소진되지 않도록 검증 후 차감
        try:
            usage_service = get_usage_limit_service()
            usage_result = usage_service.check_and_increment_usage(trimmed_device_id)

            if not usage_result["allowed"]:
                daily_limit = settings.DAILY_SYNTHESIS_LIMIT
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "daily_limit_exceeded",
                        "message": f"오늘의 무료 합성 횟수({daily_limit}회)를 모두 사용했습니다.",
                        "daily_limit": daily_limit,
                        "used": usage_result["used"],
                        "remaining": 0,
                    },
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Usage limit check failed (blocking): {str(e)}")
            raise HTTPException(
                status_code=503,
                detail="사용량 확인 서비스에 일시적 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            )

        logger.info(f"🎨 레퍼런스 합성 요청: {gender}")

        # Get synthesis service and process
        service = get_synthesis_service()
        result = service.synthesize_with_reference(
            user_image_data=user_image_data,
            reference_image_data=reference_image_data,
            gender=gender,
        )

        processing_time = round(time.time() - start_time, 2)

        if result["success"]:
            logger.info(f"✅ 레퍼런스 합성 완료 ({processing_time}초)")
            return {
                "success": True,
                "image_base64": result["image_base64"],
                "image_format": result["image_format"],
                "message": result["message"],
                "processing_time": processing_time,
            }
        else:
            logger.warning(f"⚠️ 레퍼런스 합성 실패: {result['message']}")
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": result["message"],
                    "processing_time": processing_time,
                },
            )

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
