"""BeautyMe Integrated API endpoints

종합 뷰티 컨설팅 API
- 원스톱 뷰티 분석
- AI 상담
- 리포트 생성
"""

import time
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.logging import logger, log_structured
from core.exceptions import NoFaceDetectedException, InvalidFileFormatException

# Lazy import - service will be imported when needed (Lambda cold start optimization)
_beauty_consultant_service = None


def _get_service():
    """Lazy load beauty consultant service"""
    global _beauty_consultant_service
    if _beauty_consultant_service is None:
        from services.beauty_consultant_service import get_beauty_consultant_service

        _beauty_consultant_service = get_beauty_consultant_service()
    return _beauty_consultant_service


router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


# ========== Response Models ==========


class ProfileAnalysis(BaseModel):
    """기본 분석 결과"""

    face_shape: str = Field(..., description="얼굴형")
    personal_color: str = Field(..., description="퍼스널컬러")
    gender: str = Field(..., description="성별")
    confidence: float = Field(..., description="신뢰도")


class AnalysisDetail(BaseModel):
    """분석 상세값"""

    ita_value: float = Field(..., description="ITA 값")
    hue_value: float = Field(..., description="Hue 값")
    face_ratio: float = Field(..., description="얼굴 비율")


class HairstyleItem(BaseModel):
    """헤어스타일 추천"""

    style_name: str = Field(..., description="스타일명")
    score: Optional[float] = Field(None, description="추천 점수")
    reason: Optional[str] = Field(None, description="추천 이유")
    image_search_url: Optional[str] = Field(None, description="이미지 검색 URL")


class HairColorItem(BaseModel):
    """염색 추천"""

    name: str = Field(..., description="염색명")
    hex: str = Field(..., description="HEX 코드")
    description: str = Field(default="", description="설명")
    is_trend: bool = Field(default=False, description="트렌드 여부")


class ColorPaletteItem(BaseModel):
    """컬러 팔레트"""

    name: str = Field(..., description="컬러명")
    hex: str = Field(..., description="HEX 코드")
    description: str = Field(default="", description="설명")


class StylingTips(BaseModel):
    """스타일링 조언"""

    makeup: List[str] = Field(default=[], description="메이크업 팁")
    fashion: List[str] = Field(default=[], description="패션 팁")
    description: str = Field(default="", description="상세 설명")


class Recommendations(BaseModel):
    """통합 추천"""

    hairstyles: List[HairstyleItem] = Field(default=[], description="헤어스타일 추천")
    hair_colors: List[HairColorItem] = Field(default=[], description="염색 추천")
    color_palette: List[ColorPaletteItem] = Field(default=[], description="컬러 팔레트")


class BeautyAnalysisResponse(BaseModel):
    """종합 뷰티 분석 응답"""

    success: bool
    profile: ProfileAnalysis
    analysis: AnalysisDetail
    recommendations: Recommendations
    styling: StylingTips
    processing_time: float = Field(..., description="처리 시간 (초)")


class ConsultationRequest(BaseModel):
    """상담 요청"""

    query: str = Field(..., min_length=1, max_length=500, description="질문")
    face_shape: Optional[str] = Field(None, description="얼굴형")
    personal_color: Optional[str] = Field(None, description="퍼스널컬러")
    gender: Optional[str] = Field(None, description="성별")
    session_id: Optional[str] = Field(None, description="세션 ID")


class ConsultationResponse(BaseModel):
    """상담 응답"""

    success: bool
    message: str = Field(..., description="상담 응답")
    intent: str = Field(..., description="분석된 의도")
    suggestions: List[str] = Field(default=[], description="추천 질문")


# ========== Endpoints ==========


@router.post("/beauty/analyze", response_model=BeautyAnalysisResponse, tags=["beauty"])
@limiter.limit("10/minute")
async def analyze_beauty(
    request: Request,
    file: UploadFile = File(..., description="얼굴 사진"),
    gender: str = Form("neutral", description="성별 (male/female/neutral)"),
):
    """
    🌸 BeautyMe 종합 뷰티 분석

    한 장의 얼굴 사진으로 모든 뷰티 분석과 추천을 제공합니다.

    **분석 항목:**
    - 얼굴형 분석 (5가지: 계란형, 둥근형, 각진형, 긴형, 하트형)
    - 퍼스널컬러 진단 (4계절: 봄웜, 여름쿨, 가을웜, 겨울쿨)

    **추천 항목:**
    - 맞춤 헤어스타일 3가지
    - 추천 염색 컬러 5가지 (트렌드 포함)
    - 어울리는 컬러 팔레트 6가지

    **스타일링 조언:**
    - 메이크업 팁
    - 패션 팁

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

        if gender not in ["male", "female", "neutral"]:
            gender = "neutral"

        logger.info(f"🌸 BeautyMe 분석 시작: {file.filename}, gender={gender}")

        # Read image
        image_data = await file.read()

        if len(image_data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="파일 크기가 10MB를 초과합니다")

        log_structured(
            "beauty_analysis_start",
            {
                "filename": file.filename,
                "file_size_kb": round(len(image_data) / 1024, 2),
                "gender": gender,
            },
        )

        # Analyze
        service = _get_service()
        profile = service.analyze_full(image_data, gender)

        if profile is None:
            raise NoFaceDetectedException()

        processing_time = round(time.time() - start_time, 2)

        log_structured(
            "beauty_analysis_complete",
            {
                "face_shape": profile.face_shape,
                "personal_color": profile.personal_color,
                "processing_time": processing_time,
            },
        )

        # Build response
        return BeautyAnalysisResponse(
            success=True,
            profile=ProfileAnalysis(
                face_shape=profile.face_shape,
                personal_color=profile.personal_color,
                gender=profile.gender,
                confidence=profile.confidence,
            ),
            analysis=AnalysisDetail(
                ita_value=profile.ita_value,
                hue_value=profile.hue_value,
                face_ratio=profile.face_ratio,
            ),
            recommendations=Recommendations(
                hairstyles=[
                    HairstyleItem(
                        style_name=h.get("style_name", ""),
                        score=h.get("score"),
                        reason=h.get("reason"),
                        image_search_url=h.get("image_search_url"),
                    )
                    for h in profile.hairstyles
                ],
                hair_colors=[
                    HairColorItem(
                        name=c.get("name", ""),
                        hex=c.get("hex", "#000000"),
                        description=c.get("description", ""),
                        is_trend=c.get("is_trend", False),
                    )
                    for c in profile.hair_colors
                ],
                color_palette=[
                    ColorPaletteItem(
                        name=p.get("name", ""),
                        hex=p.get("hex", "#000000"),
                        description=p.get("description", ""),
                    )
                    for p in profile.color_palette
                ],
            ),
            styling=StylingTips(
                makeup=profile.styling_tips.get("makeup", []),
                fashion=profile.styling_tips.get("fashion", []),
                description=profile.styling_tips.get("description", ""),
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
        logger.error(f"❌ BeautyMe 분석 오류: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/beauty/consult", response_model=ConsultationResponse, tags=["beauty"])
@limiter.limit("30/minute")
async def consult_beauty(request: Request, consultation: ConsultationRequest):
    """
    💬 BeautyMe AI 상담

    뷰티 관련 질문에 AI가 맞춤 상담을 제공합니다.

    **지원 주제:**
    - 헤어스타일 추천
    - 염색 컬러 상담
    - 퍼스널컬러 조언
    - 메이크업/패션 팁
    - 트렌드 정보

    **맞춤 상담:**
    - 얼굴형, 퍼스널컬러, 성별 정보를 제공하면 더 정확한 상담 가능

    **Rate Limit:** 30 requests per minute
    """
    try:
        logger.info(f"💬 BeautyMe 상담: '{consultation.query[:50]}...'")

        service = _get_service()

        # 프로필 정보가 있으면 BeautyProfile 객체 생성
        from services.beauty_consultant_service import BeautyProfile

        profile = None
        if consultation.face_shape or consultation.personal_color:
            profile = BeautyProfile(
                face_shape=consultation.face_shape or "",
                personal_color=consultation.personal_color or "",
                gender=consultation.gender or "neutral",
                confidence=1.0,
            )

        result = service.get_consultation(
            query=consultation.query,
            profile=profile,
            session_id=consultation.session_id,
        )

        return ConsultationResponse(
            success=result["success"],
            message=result["message"],
            intent=result["intent"],
            suggestions=result.get("suggestions", []),
        )

    except Exception as e:
        logger.error(f"❌ BeautyMe 상담 오류: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"상담 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/beauty/report", tags=["beauty"])
@limiter.limit("5/minute")
async def generate_report(
    request: Request,
    file: UploadFile = File(..., description="얼굴 사진"),
    gender: str = Form("neutral", description="성별"),
    format: str = Form("markdown", description="출력 형식 (markdown/json)"),
):
    """
    📊 BeautyMe 분석 리포트 생성

    종합 분석 결과를 리포트 형식으로 제공합니다.

    **출력 형식:**
    - markdown: 마크다운 형식 텍스트
    - json: JSON 형식 데이터

    **Rate Limit:** 5 requests per minute
    """
    try:
        # File validation
        if not file.filename:
            raise HTTPException(status_code=400, detail="파일명이 없습니다")

        file_ext = file.filename.lower().split(".")[-1]
        if file_ext not in ["jpg", "jpeg", "png", "webp"]:
            raise InvalidFileFormatException()

        image_data = await file.read()

        service = _get_service()
        profile = service.analyze_full(image_data, gender)

        if profile is None:
            raise NoFaceDetectedException()

        if format == "markdown":
            report = service.generate_report(profile)
            return PlainTextResponse(content=report, media_type="text/markdown")
        else:
            return {"success": True, "report": profile.to_dict()}

    except (NoFaceDetectedException, InvalidFileFormatException) as e:
        return JSONResponse(
            status_code=400, content={"success": False, "error": str(e)}
        )
    except Exception as e:
        logger.error(f"❌ 리포트 생성 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/beauty/features", tags=["beauty"])
async def get_features():
    """
    🌸 BeautyMe 기능 소개

    BeautyMe 플랫폼에서 제공하는 모든 기능을 소개합니다.
    """
    return {
        "name": "BeautyMe",
        "version": "1.0.0",
        "description": "AI 기반 종합 뷰티 컨설팅 플랫폼",
        "features": {
            "face_analysis": {
                "name": "얼굴 분석",
                "description": "MediaPipe 기반 정밀 얼굴 분석",
                "items": [
                    "얼굴형 분류 (5가지)",
                    "퍼스널컬러 진단 (4계절)",
                    "성별 추론",
                ],
            },
            "hairstyle": {
                "name": "헤어스타일",
                "description": "ML 기반 헤어스타일 추천",
                "items": [
                    "얼굴형 맞춤 추천",
                    "AI 헤어스타일 합성",
                    "트렌드 스타일 정보",
                ],
            },
            "hair_color": {
                "name": "염색",
                "description": "퍼스널컬러 기반 염색 추천",
                "items": [
                    "맞춤 염색 컬러 추천",
                    "피해야 할 컬러 안내",
                    "2024-2025 트렌드 컬러",
                    "AI 염색 시뮬레이션",
                ],
            },
            "personal_color": {
                "name": "퍼스널컬러",
                "description": "ITA + HSV 기반 퍼스널컬러 진단",
                "items": [
                    "4계절 퍼스널컬러 진단",
                    "컬러 팔레트 제공",
                    "메이크업/패션 스타일링 팁",
                ],
            },
            "chatbot": {
                "name": "AI 상담",
                "description": "RAG 기반 뷰티 상담 챗봇",
                "items": ["자연어 질문 응답", "맞춤형 상담", "지식 베이스 검색"],
            },
        },
        "endpoints": {
            "main": "/api/beauty/analyze",
            "consult": "/api/beauty/consult",
            "report": "/api/beauty/report",
        },
    }
