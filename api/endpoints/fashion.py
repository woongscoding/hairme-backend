"""체형별 아이템 추천 엔드포인트

- POST /fashion/body-analyze: 전신 사진 → 체형 분류 → 추천 핏/제휴 아이템

헤어스타일 분석과 독립된 기능이다. 무료(크레딧·Gemini 호출 없음)이고
MediaPipe Pose 만 쓴다. 사진은 메모리에서만 처리하고 저장하지 않으며,
로그에는 체형·비율·성별·처리 시간만 남긴다.

items 의 affiliate_url 은 POST /products/click (source=body_analysis)으로 발급한다.
"""

import time
from typing import Optional

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

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

_ERROR_TYPES = {
    FullBodyNotDetectedException: "full_body_not_detected",
    NotFrontalPoseException: "not_frontal_pose",
    InvalidFileFormatException: "invalid_file_format",
}


@router.post("/fashion/body-analyze")
@limiter.limit("10/minute")
async def analyze_body(
    request: Request,
    file: UploadFile = File(..., description="정면 전신 사진 (머리부터 발끝까지)"),
    gender: str = Form("male", pattern="^(male|female)$"),
    user_id: Optional[str] = Depends(get_optional_user_id),
):
    """
    전신 사진으로 체형을 분류하고 어울리는 핏/아이템을 추천 (로그인 불필요, 무료)

    체형: broad_shoulder(어깨 넓은형) / lower_volume(하체 볼륨형) / balanced(균형형)
    보조 표시(secondary): long_torso(상체 긴 비율)

    items 는 활성 제휴 제품만 포함하며 affiliate_url 은 없다 -
    POST /products/click 으로 발급받을 것. disclosure 는 반드시 함께 표시해야 한다.
    """
    start_time = time.time()

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
                "items_count": len(result["items"]),
                "authenticated": user_id is not None,
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
        log_structured(
            "body_analysis_error",
            {
                "endpoint": "fashion/body-analyze",
                "error_type": _ERROR_TYPES[type(e)],
                "status_code": 400,
                "gender": gender,
                "authenticated": user_id is not None,
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
