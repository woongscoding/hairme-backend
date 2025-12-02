"""Face analysis and hairstyle recommendation endpoints (ML-only mode)"""

import os
import time
import urllib.parse
from typing import Optional, Dict, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from services.face_detection_service import FaceDetectionService
    from services.hybrid_recommender import MLRecommendationService
    from models.mediapipe_analyzer import MediaPipeFaceFeatures

from models.mediapipe_analyzer import MediaPipeFaceFeatures

from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Depends, Form
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from config.settings import settings
from core.logging import logger, log_structured
from core.exceptions import (
    NoFaceDetectedException,
    MultipleFacesException,
    InvalidFileFormatException
)
from core.cache import calculate_image_hash, get_cached_result, save_to_cache
from models.ml_recommender import (
    predict_ml_score,
    get_confidence_level,
    get_ml_recommender
)
from core.dependencies import (
    get_face_detection_service,
    get_hybrid_service
)


router = APIRouter()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


# ========== Helper Functions ==========
def save_to_database(
    image_hash: str,
    analysis_result: Dict[str, Any],
    processing_time: float,
    detection_method: str,
    mp_features: Optional[MediaPipeFaceFeatures] = None
) -> Optional[Union[int, str]]:
    """
    Save analysis result to database using Repository pattern

    Automatically routes to MySQL or DynamoDB based on USE_DYNAMODB env variable.

    Args:
        image_hash: SHA256 hash of the image
        analysis_result: Analysis result dictionary
        processing_time: Processing time in seconds
        detection_method: Detection method used
        mp_features: MediaPipe features (optional)

    Returns:
        Record ID if successful (int for MySQL, str for DynamoDB), None otherwise
    """
    from database.repository import get_repository

    try:
        repo = get_repository()
        analysis_id = repo.save_analysis(
            image_hash=image_hash,
            analysis_result=analysis_result,
            processing_time=processing_time,
            detection_method=detection_method,
            mp_features=mp_features
        )

        # Log the result
        if analysis_id:
            backend = "dynamodb" if os.getenv('USE_DYNAMODB', 'false').lower() == 'true' else "mysql"
            recommendations = analysis_result.get("recommendations", [])

            # ML-only 모드에서는 항상 MediaPipe 결과를 사용하므로 agreement는 항상 True
            mediapipe_agreement = mp_features is not None

            log_structured("database_saved", {
                "backend": backend,
                "analysis_id": analysis_id,
                "mediapipe_enabled": mp_features is not None,
                "mediapipe_agreement": mediapipe_agreement,
                "recommendations_count": len(recommendations)
            })

        return analysis_id

    except Exception as e:
        logger.error(f"❌ 데이터베이스 저장 실패: {str(e)}")
        return None


# ========== API Endpoints ==========
@router.post("/analyze")
@limiter.limit("10/minute")  # 분당 10회 제한
async def analyze_face(
    request: Request,
    file: UploadFile = File(...),
    gender: str = Form("neutral"),  # 성별 파라미터 추가
    face_detector: 'FaceDetectionService' = Depends(get_face_detection_service),
    ml_recommender: 'MLRecommendationService' = Depends(get_hybrid_service)
):
    """
    ML 기반 얼굴 분석 및 헤어스타일 추천 (v21.0.0: ML-only mode)

    Gemini 의존성 제거 - MediaPipe + ML 모델만 사용
    """
    start_time = time.time()
    image_hash = None

    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="파일명이 없습니다")

        file_ext = file.filename.lower().split('.')[-1]
        if file_ext not in ['jpg', 'jpeg', 'png', 'webp']:
            raise InvalidFileFormatException()

        logger.info(f"🎨 ML 분석 시작: {file.filename}, gender={gender}")

        image_data = await file.read()
        image_hash = calculate_image_hash(image_data)

        log_structured("analysis_start", {
            "filename": file.filename,
            "file_size_kb": round(len(image_data) / 1024, 2),
            "image_hash": image_hash[:16],
            "method": "ml_only"
        })

        # Check cache
        cached_result = get_cached_result(image_hash)
        if cached_result:
            total_time = round(time.time() - start_time, 2)
            return {
                "success": True,
                "data": cached_result,
                "processing_time": total_time,
                "cached": True,
                "method": "ml"
            }

        # Face detection using MediaPipe
        face_detection_start = time.time()
        face_result = face_detector.detect_face(image_data)
        face_detection_time = round((time.time() - face_detection_start) * 1000, 2)

        if not face_result["has_face"]:
            log_structured("analysis_error", {
                "error_type": "no_face_detected",
                "image_hash": image_hash[:16]
            })
            raise NoFaceDetectedException()

        if face_result["face_count"] > 1:
            raise MultipleFacesException(face_count=face_result["face_count"])

        # Extract MediaPipe features
        mp_features = face_result.get("features", None)
        if not mp_features:
            raise HTTPException(
                status_code=500,
                detail="MediaPipe 얼굴 분석에 실패했습니다."
            )

        face_shape = mp_features.face_shape
        skin_tone = mp_features.skin_tone
        detected_gender = mp_features.gender if hasattr(mp_features, 'gender') else gender

        # MediaPipe 실제 측정값 추출
        face_features = getattr(mp_features, 'face_features', None)
        skin_features = getattr(mp_features, 'skin_features', None)

        logger.info(f"✅ MediaPipe 분석: {face_shape} / {skin_tone} / 성별: {detected_gender}")

        # ML 기반 추천
        ml_start = time.time()
        recommendation_result = ml_recommender.recommend(
            image_data=image_data,
            face_shape=face_shape,
            skin_tone=skin_tone,
            face_features=face_features,
            skin_features=skin_features,
            gender=gender if gender != "neutral" else detected_gender
        )
        ml_time = round((time.time() - ml_start) * 1000, 2)

        # 분석 결과 구성
        analysis_result = {
            "analysis": {
                "face_shape": face_shape,
                "personal_color": skin_tone,
                "gender": detected_gender,
                "features": f"ML 모델 기반 분석 ({face_shape}, {skin_tone})"
            },
            "recommendations": recommendation_result.get("recommendations", [])
        }

        # Naver 검색 URL 추가
        for rec in analysis_result["recommendations"]:
            style_name = rec.get("style_name", "")
            if gender == "male":
                search_query = f"남자 {style_name} 헤어스타일"
            elif gender == "female":
                search_query = f"여자 {style_name} 헤어스타일"
            else:
                search_query = f"{style_name} 헤어스타일"
            encoded_query = urllib.parse.quote(search_query)
            rec["image_search_url"] = f"https://search.naver.com/search.naver?where=image&query={encoded_query}"

        # Cache result
        save_to_cache(image_hash, analysis_result)

        # Save to database
        total_time = round(time.time() - start_time, 2)
        analysis_id = save_to_database(
            image_hash=image_hash,
            analysis_result=analysis_result,
            processing_time=total_time,
            detection_method="ml",
            mp_features=mp_features
        )

        if analysis_id is None:
            logger.warning(
                "⚠️ 데이터베이스 저장 실패 - 분석은 성공했지만 피드백을 저장할 수 없습니다. "
                f"image_hash: {image_hash[:16]}"
            )

        log_structured("analysis_complete", {
            "image_hash": image_hash[:16],
            "processing_time": total_time,
            "face_detection_time_ms": face_detection_time,
            "ml_inference_time_ms": ml_time,
            "method": "ml_only",
            "face_shape": face_shape,
            "personal_color": skin_tone,
            "analysis_id": analysis_id
        })

        return {
            "success": True,
            "data": analysis_result,
            "analysis_id": analysis_id,
            "processing_time": total_time,
            "performance": {
                "face_detection_ms": face_detection_time,
                "ml_inference_ms": ml_time,
                "detection_method": "mediapipe"
            },
            "cached": False,
            "method": "ml",
            "feedback_enabled": analysis_id is not None
        }

    except (NoFaceDetectedException, MultipleFacesException, InvalidFileFormatException) as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": e.__class__.__name__.replace("Exception", "").lower(),
                "message": str(e)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        logger.error(f"분석 중 오류 발생: {str(e)}\n{tb_str}")

        log_structured("analysis_error", {
            "error_type": "internal_error",
            "error_message": str(e),
            "image_hash": image_hash[:16] if image_hash else "unknown"
        })

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "internal_error",
                "message": f"분석 중 오류가 발생했습니다: {str(e)}",
                "traceback": tb_str
            }
        )


@router.post("/v2/analyze-hybrid")
@limiter.limit("10/minute")  # 분당 10회 제한
async def analyze_face_hybrid(
    request: Request,
    file: UploadFile = File(...),
    gender: str = Form("male"),  # 성별 파라미터 추가 (기본값: male)
    face_detector: 'FaceDetectionService' = Depends(get_face_detection_service),
    ml_recommender: 'MLRecommendationService' = Depends(get_hybrid_service)
):
    """
    ML 기반 헤어스타일 추천 (v2)

    Flow:
    1. MediaPipe로 얼굴형 + 피부톤 분석
    2. ML 모델로 Top-3 헤어스타일 추천 (성별 필터링 적용)
    """
    start_time = time.time()

    try:
        # File validation
        if not file.filename:
            raise HTTPException(status_code=400, detail="파일명이 없습니다")

        file_ext = file.filename.lower().split('.')[-1]
        if file_ext not in ['jpg', 'jpeg', 'png', 'webp']:
            raise InvalidFileFormatException()

        logger.info(f"🎨 하이브리드 분석 시작: {file.filename}, gender={gender}")

        # Read image
        image_data = await file.read()
        image_hash = calculate_image_hash(image_data)

        # 디버깅: 이미지 해시 로깅 (다른 사진인데 같은 해시가 나오는지 확인)
        logger.info(f"[IMAGE HASH] {image_hash[:16]}... (size: {len(image_data)} bytes)")

        # 1. Face detection using injected service
        import time as time_module
        face_detection_start = time_module.time()
        face_result = face_detector.detect_face(image_data)
        face_detection_time = time_module.time() - face_detection_start
        logger.info(f"[TIMING] Face detection: {face_detection_time:.2f}s")

        if not face_result["has_face"]:
            raise NoFaceDetectedException()

        mp_features = face_result.get("features")
        if not mp_features:
            raise HTTPException(
                status_code=500,
                detail="MediaPipe 얼굴 분석에 실패했습니다."
            )

        face_shape = mp_features.face_shape
        skin_tone = mp_features.skin_tone

        # MediaPipe 실제 측정값 추출 (ML 모델 입력용)
        face_features = getattr(mp_features, 'face_features', None)
        skin_features = getattr(mp_features, 'skin_features', None)

        # Null 체크 및 경고
        if face_features is None or skin_features is None:
            logger.warning(
                "⚠️ MediaPipe 측정값이 누락되었습니다. "
                "라벨 기반 추천으로 fallback합니다. "
                "ML 모델의 개인화 성능이 제한될 수 있습니다."
            )

        logger.info(f"✅ MediaPipe 분석: {face_shape} + {skin_tone}")
        if face_features is not None and skin_features is not None:
            logger.debug(f"  Face features (측정값): {face_features}")
            logger.debug(f"  Skin features (측정값): {skin_features}")

        # 2. ML recommendation using injected service
        ml_start = time_module.time()
        recommendation_result = ml_recommender.recommend(
            image_data=image_data,
            face_shape=face_shape,
            skin_tone=skin_tone,
            face_features=face_features,
            skin_features=skin_features,
            gender=gender
        )
        ml_time = time_module.time() - ml_start
        logger.info(f"[TIMING] ML recommendation: {ml_time:.2f}s")

        # 3. Add Naver search URLs (with gender prefix for better results)
        logger.info(f"[SEARCH URL] Adding search URLs with gender={gender}")
        for idx, rec in enumerate(recommendation_result.get("recommendations", [])):
            style_name = rec.get("style_name", "")

            # 성별 접두사 추가 (남성용/여성용 헤어스타일 구분)
            if gender == "male":
                search_query = f"남자 {style_name} 헤어스타일"
            elif gender == "female":
                search_query = f"여자 {style_name} 헤어스타일"
            else:
                # neutral이거나 성별 미제공 시 성별 없이 검색
                search_query = f"{style_name} 헤어스타일"

            # 첫 번째 추천 스타일의 검색어를 로깅
            if idx == 0:
                logger.info(f"[SEARCH URL DEBUG] First style: '{style_name}' -> query: '{search_query}'")

            encoded_query = urllib.parse.quote(search_query)
            rec["image_search_url"] = f"https://search.naver.com/search.naver?where=image&query={encoded_query}"

        # 4. Save to database using Repository pattern
        total_time = round(time.time() - start_time, 2)

        # Build analysis result dict for Repository
        analysis_result_for_db = {
            "analysis": {
                "face_shape": face_shape,
                "personal_color": skin_tone,
                "features": f"MediaPipe로 분석된 {face_shape}, {skin_tone} 특징"
            },
            "recommendations": recommendation_result.get("recommendations", [])
        }

        analysis_id = save_to_database(
            image_hash=image_hash,
            analysis_result=analysis_result_for_db,
            processing_time=total_time,
            detection_method="ml",
            mp_features=mp_features
        )

        # Warn if database save failed but continue with response
        if analysis_id is None:
            logger.warning(
                "⚠️ 데이터베이스 저장 실패 - 분석은 성공했지만 피드백을 저장할 수 없습니다. "
                f"image_hash: {image_hash[:16]}"
            )
            log_structured("database_save_failed", {
                "image_hash": image_hash[:16],
                "face_shape": face_shape,
                "skin_tone": skin_tone,
                "method": "ml",
                "warning": "feedback_disabled"
            })

        logger.info(f"✅ ML 분석 완료 ({total_time}초)")

        return {
            "success": True,
            "analysis_id": analysis_id,
            "data": recommendation_result,
            "processing_time": total_time,
            "method": "ml",
            "mediapipe_features": {
                "face_shape": face_shape,
                "skin_tone": skin_tone,
                "confidence": mp_features.confidence
            },
            "model_used": "hairstyle_recommender_v5_normalized.pt",
            "feedback_enabled": analysis_id is not None
        }

    except (NoFaceDetectedException, InvalidFileFormatException) as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": e.__class__.__name__.replace("Exception", "").lower(),
                "message": str(e)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ ML 분석 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"분석 중 오류가 발생했습니다: {str(e)}"
        )


# NOTE: /v2/feedback 엔드포인트는 api/endpoints/feedback.py로 이동됨
