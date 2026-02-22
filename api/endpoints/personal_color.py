"""Personal Color Analysis API endpoints

Phase 2: 퍼스널컬러 분석 API
- 이미지 기반 퍼스널컬러 진단
- 컬러 팔레트 조회
- 스타일링 조언
"""

import time
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.logging import logger, log_structured
from core.exceptions import NoFaceDetectedException, InvalidFileFormatException

# Lazy import - service will be imported when needed (Lambda cold start optimization)
_personal_color_service = None


def _get_service():
    """Lazy load personal color service"""
    global _personal_color_service
    if _personal_color_service is None:
        from services.personal_color_service import get_personal_color_service

        _personal_color_service = get_personal_color_service()
    return _personal_color_service


router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


# ========== Response Models ==========


class ColorItem(BaseModel):
    """컬러 아이템"""

    name: str = Field(..., description="컬러명")
    hex: str = Field(..., description="HEX 코드")
    description: str = Field(default="", description="설명")


class HairColorItem(BaseModel):
    """염색 추천 아이템"""

    name: str = Field(..., description="염색명")
    hex: str = Field(..., description="HEX 코드")
    description: str = Field(default="", description="설명")


class AnalysisDetail(BaseModel):
    """분석 상세"""

    ita_value: float = Field(..., description="ITA 값 (피부 밝기)")
    hue_value: float = Field(..., description="Hue 값 (색조)")
    brightness: str = Field(..., description="밝기 (bright/medium/muted)")
    undertone: str = Field(..., description="언더톤 (yellow/pink/golden/blue)")


class ColorPalette(BaseModel):
    """컬러 팔레트"""

    best_colors: List[ColorItem] = Field(default=[], description="추천 컬러")
    avoid_colors: List[str] = Field(default=[], description="피해야 할 컬러")
    hair_colors: List[HairColorItem] = Field(default=[], description="추천 염색")


class StylingAdvice(BaseModel):
    """스타일링 조언"""

    makeup_tips: List[str] = Field(default=[], description="메이크업 팁")
    fashion_tips: List[str] = Field(default=[], description="패션 팁")
    description: str = Field(default="", description="상세 설명")


class PersonalColorResponse(BaseModel):
    """퍼스널컬러 분석 응답"""

    success: bool
    personal_color: str = Field(
        ..., description="퍼스널컬러 (봄웜/여름쿨/가을웜/겨울쿨)"
    )
    confidence: float = Field(..., description="신뢰도 (0.0~1.0)")
    season: str = Field(..., description="계절 (spring/summer/autumn/winter)")
    tone: str = Field(..., description="톤 (warm/cool)")
    analysis: AnalysisDetail
    characteristics: List[str] = Field(default=[], description="특징")
    palette: ColorPalette
    styling: StylingAdvice
    processing_time: float = Field(..., description="처리 시간 (초)")


class PaletteResponse(BaseModel):
    """컬러 팔레트 조회 응답"""

    success: bool
    personal_color: str
    colors: List[ColorItem]


class StylingResponse(BaseModel):
    """스타일링 조언 응답"""

    success: bool
    personal_color: str
    makeup_tips: List[str]
    fashion_tips: List[str]
    description: str


# ========== Endpoints ==========


@router.post(
    "/personal-color", response_model=PersonalColorResponse, tags=["personal_color"]
)
@limiter.limit("10/minute")
async def analyze_personal_color(
    request: Request, file: UploadFile = File(..., description="얼굴 이미지 파일")
):
    """
    퍼스널컬러 분석

    얼굴 이미지를 분석하여 퍼스널컬러를 진단합니다.

    **분석 항목:**
    - 퍼스널컬러 진단 (봄웜/여름쿨/가을웜/겨울쿨)
    - ITA 기반 피부 밝기 분석
    - 언더톤 분석 (웜/쿨)

    **제공 정보:**
    - 추천 컬러 팔레트 (HEX 코드 포함)
    - 피해야 할 컬러
    - 추천 염색 컬러
    - 메이크업 & 패션 스타일링 팁

    **Rate Limit:** 10 requests per minute
    """
    start_time = time.time()

    try:
        # File validation
        if not file.filename:
            raise HTTPException(status_code=400, detail="파일명이 없습니다")

        file_ext = file.filename.lower().split(".")[-1]
        if file_ext not in ["jpg", "jpeg", "png", "webp"]:
            raise InvalidFileFormatException()

        logger.info(f"🎨 퍼스널컬러 분석 시작: {file.filename}")

        # Read image
        image_data = await file.read()

        log_structured(
            "personal_color_start",
            {
                "filename": file.filename,
                "file_size_kb": round(len(image_data) / 1024, 2),
            },
        )

        # Analyze
        service = _get_service()
        result = service.analyze(image_data)

        if result is None:
            raise NoFaceDetectedException()

        processing_time = round(time.time() - start_time, 2)

        log_structured(
            "personal_color_complete",
            {
                "personal_color": result.personal_color,
                "confidence": result.confidence,
                "processing_time": processing_time,
            },
        )

        # Build response
        return PersonalColorResponse(
            success=True,
            personal_color=result.personal_color,
            confidence=result.confidence,
            season=result.season,
            tone=result.tone,
            analysis=AnalysisDetail(
                ita_value=result.ita_value,
                hue_value=result.hue_value,
                brightness=result.brightness,
                undertone=result.undertone,
            ),
            characteristics=result.characteristics,
            palette=ColorPalette(
                best_colors=[ColorItem(**c) for c in result.best_colors],
                avoid_colors=result.avoid_colors,
                hair_colors=[HairColorItem(**c) for c in result.hair_colors],
            ),
            styling=StylingAdvice(
                makeup_tips=result.makeup_tips,
                fashion_tips=result.fashion_tips,
                description=result.styling_description,
            ),
            processing_time=processing_time,
        )

    except (NoFaceDetectedException, InvalidFileFormatException) as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": e.__class__.__name__.replace("Exception", "").lower(),
                "message": str(e),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 퍼스널컬러 분석 오류: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}"
        )


@router.get(
    "/personal-color/{color_type}/palette",
    response_model=PaletteResponse,
    tags=["personal_color"],
)
async def get_color_palette(color_type: str):
    """
    특정 퍼스널컬러의 컬러 팔레트 조회

    **Path Parameters:**
    - color_type: 퍼스널컬러 타입 (봄웜, 여름쿨, 가을웜, 겨울쿨)

    **Returns:**
    - 추천 컬러 목록 (이름, HEX 코드, 설명)
    """
    valid_types = ["봄웜", "여름쿨", "가을웜", "겨울쿨"]
    if color_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"유효하지 않은 퍼스널컬러입니다. 가능한 값: {valid_types}",
        )

    service = _get_service()
    colors = service.get_color_palette(color_type)

    return PaletteResponse(
        success=True, personal_color=color_type, colors=[ColorItem(**c) for c in colors]
    )


@router.get(
    "/personal-color/{color_type}/styling",
    response_model=StylingResponse,
    tags=["personal_color"],
)
async def get_styling_tips(color_type: str):
    """
    특정 퍼스널컬러의 스타일링 조언 조회

    **Path Parameters:**
    - color_type: 퍼스널컬러 타입 (봄웜, 여름쿨, 가을웜, 겨울쿨)

    **Returns:**
    - 메이크업 팁
    - 패션 팁
    - 상세 설명
    """
    valid_types = ["봄웜", "여름쿨", "가을웜", "겨울쿨"]
    if color_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"유효하지 않은 퍼스널컬러입니다. 가능한 값: {valid_types}",
        )

    service = _get_service()
    tips = service.get_styling_tips(color_type)

    return StylingResponse(
        success=True,
        personal_color=color_type,
        makeup_tips=tips.get("makeup_tips", []),
        fashion_tips=tips.get("fashion_tips", []),
        description=tips.get("description", ""),
    )


@router.get("/personal-color/{color_type}/hair", tags=["personal_color"])
async def get_hair_recommendations(color_type: str):
    """
    특정 퍼스널컬러의 염색 추천 조회

    **Path Parameters:**
    - color_type: 퍼스널컬러 타입 (봄웜, 여름쿨, 가을웜, 겨울쿨)

    **Returns:**
    - 추천 염색 컬러 목록
    - 피해야 할 염색 컬러
    """
    valid_types = ["봄웜", "여름쿨", "가을웜", "겨울쿨"]
    if color_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"유효하지 않은 퍼스널컬러입니다. 가능한 값: {valid_types}",
        )

    service = _get_service()
    hair = service.get_hair_recommendations(color_type)

    return {
        "success": True,
        "personal_color": color_type,
        "recommended": hair.get("recommended", []),
        "avoid": hair.get("avoid", []),
    }


@router.get("/personal-color/types", tags=["personal_color"])
async def get_all_types():
    """
    지원하는 모든 퍼스널컬러 타입 조회

    **Returns:**
    - 4계절 퍼스널컬러 정보
    """
    service = _get_service()

    types = []
    for pc_type in ["봄웜", "여름쿨", "가을웜", "겨울쿨"]:
        info = service.personal_color_data.get(pc_type, {})
        types.append(
            {
                "type": pc_type,
                "korean_name": info.get("korean_name", pc_type),
                "english_name": info.get("english_name", ""),
                "season": info.get("season", ""),
                "tone": info.get("tone", ""),
                "description": info.get("description", "")[:100] + "...",
            }
        )

    return {"success": True, "types": types}
