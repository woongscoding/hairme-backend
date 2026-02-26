"""Hair Color Recommendation and Synthesis API endpoints

Phase 3: 염색 추천 + 합성 API
- 퍼스널컬러 기반 염색 추천
- 트렌드 염색 컬러 조회
- 가상 염색 시뮬레이션
"""

import time
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.logging import logger, log_structured
from core.exceptions import InvalidFileFormatException
from services.usage_limit_service import get_usage_limit_service
from config.settings import settings

# Lazy import - service will be imported when needed (Lambda cold start optimization)
_hair_color_service = None


def _get_service():
    """Lazy load hair color service"""
    global _hair_color_service
    if _hair_color_service is None:
        from services.hair_color_service import get_hair_color_service

        _hair_color_service = get_hair_color_service()
    return _hair_color_service


router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


# ========== Response Models ==========


class HairColorItem(BaseModel):
    """염색 컬러 아이템"""

    name: str = Field(..., description="염색명")
    hex: str = Field(..., description="HEX 코드")
    level: str = Field(..., description="밝기 레벨")
    description: str = Field(default="", description="설명")
    suitable_for: List[str] = Field(default=[], description="적합한 대상")
    is_trend: bool = Field(default=False, description="트렌드 여부")


class AvoidColorItem(BaseModel):
    """피해야 할 컬러"""

    name: str = Field(..., description="염색명")
    reason: str = Field(..., description="피해야 할 이유")


class HairColorRecommendationResponse(BaseModel):
    """염색 추천 응답"""

    success: bool
    personal_color: str = Field(..., description="퍼스널컬러")
    recommended: List[HairColorItem] = Field(default=[], description="추천 컬러")
    avoid: List[AvoidColorItem] = Field(default=[], description="피해야 할 컬러")
    trends: List[HairColorItem] = Field(default=[], description="트렌드 컬러")


class SynthesisResponse(BaseModel):
    """염색 시뮬레이션 응답"""

    success: bool
    image_base64: Optional[str] = Field(None, description="Base64 인코딩된 결과 이미지")
    image_format: Optional[str] = Field(None, description="이미지 포맷")
    message: str = Field(..., description="결과 메시지")
    color_name: str = Field(..., description="적용된 염색명")
    color_hex: str = Field(..., description="적용된 HEX 코드")
    processing_time: float = Field(..., description="처리 시간 (초)")


# ========== Endpoints ==========


@router.get(
    "/hair-color/{personal_color}",
    response_model=HairColorRecommendationResponse,
    tags=["hair_color"],
)
async def get_hair_color_recommendations(
    personal_color: str, include_trends: bool = True
):
    """
    퍼스널컬러 기반 염색 추천

    퍼스널컬러에 맞는 추천 염색 컬러와 피해야 할 컬러를 반환합니다.

    **Path Parameters:**
    - personal_color: 퍼스널컬러 타입 (봄웜/여름쿨/가을웜/겨울쿨)

    **Query Parameters:**
    - include_trends: 트렌드 컬러 포함 여부 (기본값: true)

    **Returns:**
    - 추천 염색 컬러 목록
    - 피해야 할 컬러 목록
    - 트렌드 염색 컬러
    """
    valid_types = ["봄웜", "여름쿨", "가을웜", "겨울쿨"]
    if personal_color not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"유효하지 않은 퍼스널컬러입니다. 가능한 값: {valid_types}",
        )

    service = _get_service()
    result = service.get_recommendations(personal_color, include_trends)

    return HairColorRecommendationResponse(
        success=True,
        personal_color=result.personal_color,
        recommended=[
            HairColorItem(
                name=c.name,
                hex=c.hex,
                level=c.level,
                description=c.description,
                suitable_for=c.suitable_for,
                is_trend=c.is_trend,
            )
            for c in result.recommended
        ],
        avoid=[
            AvoidColorItem(name=a["name"], reason=a["reason"]) for a in result.avoid
        ],
        trends=[
            HairColorItem(
                name=c.name,
                hex=c.hex,
                level=c.level,
                description=c.description,
                suitable_for=c.suitable_for,
                is_trend=True,
            )
            for c in result.trends
        ],
    )


@router.get("/hair-color/trends/all", tags=["hair_color"])
async def get_all_trend_colors():
    """
    모든 트렌드 염색 컬러 조회

    시즌별 트렌드 염색 컬러를 반환합니다.

    **Returns:**
    - 시즌별 트렌드 컬러 목록 (2024_winter, 2025_spring 등)
    """
    service = _get_service()
    trends = service.get_all_trends()

    return {"success": True, "trends": trends}


@router.get("/hair-color/search/{color_name}", tags=["hair_color"])
async def search_color(color_name: str):
    """
    염색 컬러명으로 검색

    **Path Parameters:**
    - color_name: 검색할 염색 컬러명

    **Returns:**
    - 컬러 정보 (이름, HEX, 레벨, 설명, 적합한 퍼스널컬러)
    """
    service = _get_service()
    color = service.get_color_by_name(color_name)

    if color is None:
        raise HTTPException(
            status_code=404, detail=f"'{color_name}' 컬러를 찾을 수 없습니다."
        )

    return {"success": True, "color": color}


@router.post(
    "/hair-color/synthesize", response_model=SynthesisResponse, tags=["hair_color"]
)
@limiter.limit("5/minute")
async def synthesize_hair_color(
    request: Request,
    file: UploadFile = File(..., description="사용자 얼굴 사진"),
    color_name: str = Form(..., description="염색 컬러명 (예: 밀크브라운)"),
    color_hex: str = Form(None, description="HEX 코드 (선택, 미입력시 자동 조회)"),
    device_id: str = Form(
        ..., description="디바이스 고유 ID (일일 사용량 제한용)"
    ),
    additional_instructions: Optional[str] = Form(None, description="추가 요청사항"),
):
    """
    가상 염색 시뮬레이션 API

    사용자 사진에 선택한 염색 컬러를 적용한 이미지를 생성합니다.

    **Form Data:**
    - file: 사용자 얼굴 사진 (JPG, PNG, WEBP)
    - color_name: 염색 컬러명 (예: "밀크브라운", "애쉬브라운")
    - color_hex: HEX 코드 (선택, 미입력시 자동 조회)
    - additional_instructions: 추가 스타일링 요청 (선택)

    **Returns:**
    - Base64 인코딩된 결과 이미지
    - 처리 시간

    **Rate Limit:** 5 requests per minute
    """
    start_time = time.time()

    try:
        # Daily usage limit: atomic check + increment (server-side enforcement)
        trimmed_device_id = device_id.strip()
        if not trimmed_device_id:
            raise HTTPException(status_code=400, detail="device_id는 필수입니다.")

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

        # File validation
        if not file.filename:
            raise HTTPException(status_code=400, detail="파일명이 없습니다")

        file_ext = file.filename.lower().split(".")[-1]
        if file_ext not in ["jpg", "jpeg", "png", "webp"]:
            raise InvalidFileFormatException()

        logger.info(f"🎨 염색 시뮬레이션 요청: {color_name}")

        # Read image
        image_data = await file.read()

        # File size validation
        if len(image_data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="파일 크기가 10MB를 초과합니다")

        # If HEX not provided, look up by color name
        service = _get_service()
        if not color_hex:
            color_info = service.get_color_by_name(color_name)
            if color_info:
                color_hex = color_info.get("hex", "#8B4513")  # Default brown
                logger.info(f"컬러 조회 성공: {color_name} -> {color_hex}")
            else:
                color_hex = "#8B4513"  # Default brown if not found
                logger.warning(f"컬러 미발견, 기본값 사용: {color_name} -> {color_hex}")

        log_structured(
            "hair_color_synthesis_start",
            {
                "color_name": color_name,
                "color_hex": color_hex,
                "file_size_kb": round(len(image_data) / 1024, 2),
            },
        )

        # Synthesize
        result = service.synthesize_hair_color(
            image_data=image_data,
            color_name=color_name,
            color_hex=color_hex,
            additional_instructions=additional_instructions,
        )

        processing_time = round(time.time() - start_time, 2)

        if result["success"]:
            log_structured(
                "hair_color_synthesis_success",
                {"color_name": color_name, "processing_time": processing_time},
            )

            return SynthesisResponse(
                success=True,
                image_base64=result["image_base64"],
                image_format=result["image_format"],
                message=result["message"],
                color_name=color_name,
                color_hex=color_hex,
                processing_time=processing_time,
            )
        else:
            log_structured(
                "hair_color_synthesis_failed",
                {"color_name": color_name, "message": result["message"]},
            )

            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": result["message"],
                    "color_name": color_name,
                    "color_hex": color_hex,
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
        logger.error(f"❌ 염색 시뮬레이션 오류: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"염색 시뮬레이션 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/hair-color/synthesize-by-personal-color", tags=["hair_color"])
@limiter.limit("5/minute")
async def synthesize_recommended_color(
    request: Request,
    file: UploadFile = File(..., description="사용자 얼굴 사진"),
    personal_color: str = Form(
        ..., description="퍼스널컬러 (봄웜/여름쿨/가을웜/겨울쿨)"
    ),
    color_index: int = Form(0, description="추천 컬러 인덱스 (0: 첫번째 추천)"),
    device_id: str = Form(
        ..., description="디바이스 고유 ID (일일 사용량 제한용)"
    ),
):
    """
    퍼스널컬러 기반 추천 염색 시뮬레이션

    퍼스널컬러에 맞는 추천 염색을 자동으로 적용합니다.

    **Form Data:**
    - file: 사용자 얼굴 사진
    - personal_color: 퍼스널컬러 (봄웜/여름쿨/가을웜/겨울쿨)
    - color_index: 추천 컬러 중 적용할 인덱스 (0부터 시작)

    **Returns:**
    - 염색 적용된 이미지
    - 적용된 컬러 정보

    **Rate Limit:** 5 requests per minute
    """
    start_time = time.time()

    try:
        # Daily usage limit: atomic check + increment (server-side enforcement)
        trimmed_device_id = device_id.strip()
        if not trimmed_device_id:
            raise HTTPException(status_code=400, detail="device_id는 필수입니다.")

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

        valid_types = ["봄웜", "여름쿨", "가을웜", "겨울쿨"]
        if personal_color not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"유효하지 않은 퍼스널컬러입니다. 가능한 값: {valid_types}",
            )

        # Get recommendations
        service = _get_service()
        result = service.get_recommendations(personal_color, include_trends=False)

        if color_index >= len(result.recommended):
            raise HTTPException(
                status_code=400,
                detail=f"color_index가 범위를 벗어났습니다. (최대: {len(result.recommended) - 1})",
            )

        selected_color = result.recommended[color_index]

        # File validation
        if not file.filename:
            raise HTTPException(status_code=400, detail="파일명이 없습니다")

        file_ext = file.filename.lower().split(".")[-1]
        if file_ext not in ["jpg", "jpeg", "png", "webp"]:
            raise InvalidFileFormatException()

        image_data = await file.read()

        if len(image_data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="파일 크기가 10MB를 초과합니다")

        logger.info(
            f"🎨 퍼스널컬러 기반 염색: {personal_color} -> {selected_color.name}"
        )

        # Synthesize
        synthesis_result = service.synthesize_hair_color(
            image_data=image_data,
            color_name=selected_color.name,
            color_hex=selected_color.hex,
        )

        processing_time = round(time.time() - start_time, 2)

        if synthesis_result["success"]:
            return {
                "success": True,
                "image_base64": synthesis_result["image_base64"],
                "image_format": synthesis_result["image_format"],
                "message": synthesis_result["message"],
                "personal_color": personal_color,
                "applied_color": {
                    "name": selected_color.name,
                    "hex": selected_color.hex,
                    "level": selected_color.level,
                    "description": selected_color.description,
                },
                "processing_time": processing_time,
            }
        else:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": synthesis_result["message"],
                    "personal_color": personal_color,
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
        logger.error(f"❌ 퍼스널컬러 기반 염색 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
