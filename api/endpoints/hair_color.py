"""Hair Color Recommendation and Synthesis API endpoints

Phase 3: 염색 추천 + 합성 API
- 퍼스널컬러 기반 염색 추천 (GET /hair-color/{personal_color})
- 가상 염색 시뮬레이션 (POST /hair-color/synthesize)

과금 정책 (합성 계열은 core.quota 공용 로직 사용):
- 로그인 회원 (Authorization: Bearer <JWT>): 크레딧 차감 (합성 실패 시 자동 환불)
- 비로그인: device_id 일일 무료 제한 + 클라이언트 IP 일일 상한
- 모든 입력 검증은 과금보다 먼저 수행
- 화질 등급은 스타일 합성과 같다 (services.result_quality):
  구매 크레딧으로 차감되면 1K, 그 외는 512px + 워터마크 (is_preview=true)

비용 보호 (api/endpoints/synthesis.py 와 같은 순서·같은 응답 - 공통 단계는
services/synthesis_pipeline.py):
  입력 검증 → 일 예산 상한(비로그인만, 503) → 중복 요청 잠금(409) → 과금
  → 합성 → record_api_calls(실제 Gemini 호출 수) → 실패 시 환불 → 잠금 해제
- 결과 캐시는 없다 (스타일 합성과 다른 점). 잠금 키는 같은 사진 + 같은 컬러 +
  같은 추가 요청의 해시라, 같은 요청을 연달아 누르면 두 번째는 과금 전에 409.
- 구조화 로그의 api_calls 는 Gemini 를 실제로 시도한 이벤트에만 넣는다
  (사전 거절 409/503/402/429 에는 넣지 않는다 - synthesis.py 와 같은 규약).
"""

import hashlib
import re
import time
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Request, Form
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.logging import logger, log_structured
from core.exceptions import InvalidFileFormatException
from core.jwt_auth import get_optional_user_id
from core.quota import charge_synthesis_quota, client_ip_from_request
from core.synthesis_budget import daily_budget_exceeded, record_api_calls
from core.synthesis_lock import NOOP_RELEASE, acquire_synthesis_lock
from services.result_quality import (
    charged_tier,
    finalize_image,
    image_size_for,
    is_preview,
    result_quality,
)
from services.synthesis_pipeline import (
    SynthesisPipeline,
    internal_error,
    invalid_file_format_response,
    refund_on_error,
)
from core.upload_validation import (
    MAX_ADDITIONAL_INSTRUCTIONS_LENGTH,
    MAX_HAIRSTYLE_NAME_LENGTH,
    sanitize_prompt_text,
    validate_file_extension,
    validate_image_upload,
)

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

ENDPOINT_SYNTHESIZE = "hair-color/synthesize"


def _request_key(
    image_data: bytes,
    color_name: str,
    color_hex: str,
    additional_instructions: Optional[str] = None,
) -> str:
    """중복 요청 잠금용 키 (같은 사진 + 같은 컬러 + 같은 추가 요청).

    스타일 합성의 cache_key 와 같은 방식(SHA-256, 부분마다 "|" 구분)이고,
    "hair-color" 접두 부분으로 스타일 합성 키와 섞이지 않게 한다.
    """
    h = hashlib.sha256()
    for part in (
        b"hair-color",
        image_data,
        color_name.encode("utf-8"),
        (color_hex or "").upper().lstrip("#").encode("utf-8"),
        (additional_instructions or "").encode("utf-8"),
    ):
        h.update(part)
        h.update(b"|")
    return h.hexdigest()


def _pipeline(user_id: Optional[str], start_time: float) -> SynthesisPipeline:
    """이 요청의 공통 단계 (예산·잠금·호출 집계·실패 로그 - services.synthesis_pipeline).

    예산/잠금/집계/로그 함수를 호출 시점에 이 모듈의 이름으로 넘긴다
    (api.endpoints.hair_color.* 를 patch 하는 테스트·호출부가 그대로 동작하도록).
    """
    return SynthesisPipeline(
        endpoint=ENDPOINT_SYNTHESIZE,
        failed_event="hair_color_synthesis_failed",
        user_id=user_id,
        start_time=start_time,
        log=log_structured,
        budget_exceeded=daily_budget_exceeded,
        acquire_lock=acquire_synthesis_lock,
        record_calls=record_api_calls,
    )


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
    quota: Optional[Dict[str, Any]] = Field(
        None, description="과금 정보 (크레딧 잔액 또는 무료 잔여 횟수)"
    )
    quality: Optional[Dict[str, Any]] = Field(
        None, description="화질 등급 (tier, image_size, watermarked)"
    )
    is_preview: Optional[bool] = Field(
        None, description="무료 미리보기(512px + 워터마크) 여부"
    )


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


@router.post(
    "/hair-color/synthesize", response_model=SynthesisResponse, tags=["hair_color"]
)
@limiter.limit("5/minute")
async def synthesize_hair_color(
    request: Request,
    file: UploadFile = File(..., description="사용자 얼굴 사진"),
    color_name: str = Form(..., description="염색 컬러명 (예: 밀크브라운)"),
    color_hex: str = Form(None, description="HEX 코드 (선택, 미입력시 자동 조회)"),
    device_id: Optional[str] = Form(
        None, description="디바이스 고유 ID (비로그인 일일 사용량 제한용)"
    ),
    additional_instructions: Optional[str] = Form(None, description="추가 요청사항"),
    user_id: Optional[str] = Depends(get_optional_user_id),
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

    사전 거절: 503 daily_budget_exceeded (비로그인, 일 예산 상한) /
    409 synthesis_in_progress (같은 요청 진행 중) - 둘 다 과금 전이다.
    """
    start_time = time.time()
    # 잠금을 잡기 전에 빠져나가는 경로가 있으므로 기본값은 no-op
    release_lock: Callable[[], None] = NOOP_RELEASE
    pipeline = _pipeline(user_id, start_time)

    try:
        # ========== 1. 입력 검증 (과금 전에 수행) ==========
        if not user_id and not (device_id or "").strip():
            raise HTTPException(status_code=400, detail="device_id는 필수입니다.")

        # File validation
        validate_file_extension(file.filename)

        # Gemini 프롬프트에 삽입되는 사용자 입력 정제 (프롬프트 인젝션 완화)
        color_name = sanitize_prompt_text(
            color_name, MAX_HAIRSTYLE_NAME_LENGTH, "염색 컬러명"
        )
        if not color_name:
            raise HTTPException(
                status_code=400, detail="염색 컬러명이 올바르지 않습니다."
            )
        if color_hex:
            color_hex = color_hex.strip()
            if not re.fullmatch(r"#?[0-9A-Fa-f]{6}", color_hex):
                raise HTTPException(
                    status_code=400, detail="color_hex 형식이 올바르지 않습니다."
                )
        if additional_instructions:
            additional_instructions = sanitize_prompt_text(
                additional_instructions,
                MAX_ADDITIONAL_INSTRUCTIONS_LENGTH,
                "추가 요청",
            )

        # Read image + 실제 크기/내용 검증
        image_data = await file.read()
        validate_image_upload(image_data)

        logger.info(f"🎨 염색 시뮬레이션 요청: {color_name}")

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

        # ========== 2. 일 예산 상한 + 중복 요청 잠금 (과금 전) ==========
        rejection, release_lock = pipeline.pre_charge_guard(
            device_id,
            _request_key(image_data, color_name, color_hex, additional_instructions),
        )
        if rejection is not None:
            return rejection

        # ========== 3. 과금 (회원: 크레딧 / 비로그인: device + IP 일일 제한) ==========
        quota_error, quota, refund = charge_synthesis_quota(
            user_id,
            device_id,
            client_ip_from_request(request),
            endpoint=ENDPOINT_SYNTHESIZE,
        )
        if quota_error is not None:
            return quota_error

        tier = charged_tier(quota)

        # ========== 4. 합성 (과금 이후 모든 실패 경로에서 환불 보장) ==========
        with refund_on_error(refund):
            log_structured(
                "hair_color_synthesis_start",
                {
                    "endpoint": ENDPOINT_SYNTHESIZE,
                    "mode": pipeline.mode(quota),
                    "tier": tier,
                    "color_name": color_name,
                    "color_hex": color_hex,
                    "file_size_kb": round(len(image_data) / 1024, 2),
                },
            )

            result = service.synthesize_hair_color(
                image_data=image_data,
                color_name=color_name,
                color_hex=color_hex,
                additional_instructions=additional_instructions,
                image_size=image_size_for(tier),
            )
            # 성공/실패 모두 비용은 이미 나갔으므로 일 예산 집계에 반영한다
            api_calls = pipeline.record_gemini_calls(result)
            if result["success"]:
                result["image_base64"], result["image_format"] = finalize_image(
                    result, tier
                )

        processing_time = pipeline.elapsed()

        if result["success"]:
            log_structured(
                "hair_color_synthesis_success",
                {
                    "endpoint": ENDPOINT_SYNTHESIZE,
                    "mode": pipeline.mode(quota),
                    "color_name": color_name,
                    "api_calls": api_calls,
                    "processing_time": processing_time,
                    "tier": tier,
                    "authenticated": user_id is not None,
                },
            )

            return SynthesisResponse(
                success=True,
                image_base64=result["image_base64"],
                image_format=result["image_format"],
                message=result["message"],
                color_name=color_name,
                color_hex=color_hex,
                processing_time=processing_time,
                quota=quota,
                quality=result_quality(tier),
                is_preview=is_preview(tier),
            )

        # 환불 → 실패 로그 → 422
        return pipeline.reject_synthesis(
            result,
            quota,
            refund,
            api_calls,
            processing_time,
            log_fields={
                "color_name": color_name,
                "message": result["message"],
                "failure_reason": result.get("failure_reason"),
            },
            body_extra={"color_name": color_name, "color_hex": color_hex},
        )

    except InvalidFileFormatException as e:
        return invalid_file_format_response(e)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 염색 시뮬레이션 오류: {str(e)}", exc_info=True)
        raise internal_error()
    finally:
        # 완료/실패/예외 어느 경로로 빠져나가도 잠금을 놓는다
        release_lock()
