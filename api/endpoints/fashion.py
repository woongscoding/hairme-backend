"""체형별 아이템 추천 엔드포인트

- POST /fashion/body-analyze: 전신 사진 → 체형 분류 → 추천 핏/제휴 아이템

헤어스타일 분석과 독립된 기능이다. 무료(크레딧·Gemini 호출 없음)이고
MediaPipe Pose 만 쓴다. 사진은 요청에 save_consent=true 가 있을 때만
S3 body/originals/ 에 저장한다 (헤어 사진과 경로·동의 분리, 1년 보관).
동의가 없으면 메모리에서만 처리한다. 로그에는 체형·비율·성별·처리 시간만 남긴다.

recommended[].product_id 가 있으면 POST /products/click (source=body_analysis)으로
쿠팡 검색 결과 링크(affiliate_url)를 발급한다.
"""

import re
import time
from typing import Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.exceptions import (
    FullBodyNotDetectedException,
    InvalidFileFormatException,
    NotFrontalPoseException,
)
from core.jwt_auth import get_optional_user_id
from core.logging import logger, log_structured
from core.upload_validation import validate_file_extension, validate_image_upload
from models.body_pose_analyzer import (
    classify_body_from_ratios,
    compute_body_ratios,
    get_body_pose_analyzer,
)
from services.body_fashion_service import get_body_fashion_service
from services.photo_storage_service import get_photo_storage_service

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

_ERROR_TYPES = {
    FullBodyNotDetectedException: "full_body_not_detected",
    NotFrontalPoseException: "not_frontal_pose",
    InvalidFileFormatException: "invalid_file_format",
}

_DEVICE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{8,64}")


def _photo_subject(user_id: Optional[str], device_id: Optional[str]) -> str:
    """S3 경로에 쓸 사진 주체. 회원 > 기기 > 익명 순"""
    if user_id:
        return f"user-{user_id}"
    if device_id and _DEVICE_ID_PATTERN.fullmatch(device_id):
        return f"device-{device_id}"
    return "anonymous"


async def _save_photo_if_consented(
    consent: bool,
    image_data: Optional[bytes],
    user_id: Optional[str],
    device_id: Optional[str],
    metadata: Dict[str, str],
) -> bool:
    """동의한 요청의 사진만 저장. 저장 실패는 분석 응답에 영향을 주지 않는다."""
    if not consent or not image_data:
        return False
    try:
        key = await run_in_threadpool(
            get_photo_storage_service().save_body_photo,
            _photo_subject(user_id, device_id),
            image_data,
            metadata,
        )
        return key is not None
    except Exception as e:
        logger.warning(f"체형 사진 저장 실패 (무시): {e}")
        return False


@router.post("/fashion/body-analyze")
@limiter.limit("10/minute")
async def analyze_body(
    request: Request,
    file: UploadFile = File(..., description="정면 전신 사진 (머리부터 발끝까지)"),
    gender: str = Form("male", pattern="^(male|female)$"),
    save_consent: bool = Form(
        False,
        description="사진 저장 동의 (true 일 때만 S3 body/originals/ 에 1년 보관)",
    ),
    device_id: Optional[str] = Form(
        None,
        max_length=128,
        description="비로그인 사진 주체 (저장 동의 시 경로에 사용)",
    ),
    user_id: Optional[str] = Depends(get_optional_user_id),
):
    """
    전신 사진으로 체형을 분류하고 어울리는 핏/아이템을 추천 (로그인 불필요, 무료)

    체형: broad_shoulder(어깨 넓은형) / lower_volume(하체 볼륨형) / balanced(균형형)
    보조 표시(secondary): long_torso(상체 긴 비율)

    recommended[].product_id 는 활성 쿠팡 검색 링크가 있을 때만 값이 있다 (없으면 null).
    affiliate_url 은 POST /products/click 으로 발급받을 것.
    disclosure 는 링크를 보여줄 때 반드시 함께 표시해야 한다.
    """
    start_time = time.time()
    image_data: Optional[bytes] = None

    try:
        validate_file_extension(file.filename)
        image_data = await file.read()
        validate_image_upload(image_data)

        # 모델 로드·추론은 CPU 작업이라 스레드풀로 - 이벤트 루프 차단 방지
        analyzer = await run_in_threadpool(get_body_pose_analyzer)
        pose = await run_in_threadpool(analyzer.extract_landmarks, image_data)
        if pose is None:
            raise FullBodyNotDetectedException()

        landmarks, width, height = pose
        ratios = compute_body_ratios(landmarks, width, height)
        body_type, flags = classify_body_from_ratios(
            ratios.shoulder_hip_ratio, ratios.leg_ratio, gender
        )
        result = get_body_fashion_service().build_recommendation(
            body_type, flags, gender
        )
        processing_time = round(time.time() - start_time, 2)

        photo_saved = await _save_photo_if_consented(
            save_consent,
            image_data,
            user_id,
            device_id,
            {
                "gender": gender,
                "body-type": body_type,
                "flags": ",".join(flags),
                "shoulder-hip": f"{ratios.shoulder_hip_ratio:.3f}",
                "leg": f"{ratios.leg_ratio:.3f}",
                "shoulder-torso": f"{ratios.shoulder_torso_ratio:.3f}",
            },
        )

        # 사진·식별자는 남기지 않는다 - 임계값 재보정에 쓸 비율만
        log_structured(
            "body_analysis_complete",
            {
                "endpoint": "fashion/body-analyze",
                "body_type": body_type,
                "flags": flags,
                "gender": gender,
                "shoulder_hip_ratio": round(ratios.shoulder_hip_ratio, 3),
                "leg_ratio": round(ratios.leg_ratio, 3),
                "shoulder_torso_ratio": round(ratios.shoulder_torso_ratio, 3),
                "linked_count": sum(
                    1 for fit in result["recommended"] if fit["product_id"]
                ),
                "authenticated": user_id is not None,
                "save_consent": save_consent,
                "photo_saved": photo_saved,
                "processing_time": processing_time,
            },
        )

        return {
            "success": True,
            **result,
            "gender": gender,
            "ratios": ratios.to_dict(),
            "processing_time": processing_time,
        }

    except tuple(_ERROR_TYPES) as e:
        # 전신이 안 잡히거나 정면이 아닌 사진도 동의했다면 남긴다 (가이드·검출 개선용).
        # 파일 형식 오류는 image_data 가 None 이거나 이미지가 아니라 저장하지 않는다.
        photo_saved = False
        if not isinstance(e, InvalidFileFormatException):
            photo_saved = await _save_photo_if_consented(
                save_consent,
                image_data,
                user_id,
                device_id,
                {"gender": gender, "error": _ERROR_TYPES[type(e)]},
            )
        log_structured(
            "body_analysis_error",
            {
                "endpoint": "fashion/body-analyze",
                "error_type": _ERROR_TYPES[type(e)],
                "status_code": 400,
                "gender": gender,
                "authenticated": user_id is not None,
                "save_consent": save_consent,
                "photo_saved": photo_saved,
            },
        )
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
        logger.error(f"❌ 체형 분석 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )
