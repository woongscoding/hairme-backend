"""Face analysis and hairstyle recommendation endpoints"""

import os
import time
import urllib.parse
from typing import Optional, Dict, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from services.face_detection_service import FaceDetectionService
    from services.gemini_analysis_service import GeminiAnalysisService
    from services.hybrid_recommender import MLRecommendationService
    from services.feedback_collector import FeedbackCollector
    from services.retrain_queue import RetrainQueue
    from models.mediapipe_analyzer import MediaPipeFaceFeatures

from models.mediapipe_analyzer import MediaPipeFaceFeatures  # Keep this if used at runtime?
# Wait, MediaPipeFaceFeatures is a dataclass, it might be lightweight.
# But models.mediapipe_analyzer imports logging, dataclasses, typing, math.
# It does NOT import mediapipe at top level anymore.
# So importing MediaPipeFaceFeatures is fine.

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
from core.ml_loader import (
    predict_ml_score,
    get_confidence_level,
    sentence_transformer
)
from core.dependencies import (
    get_face_detection_service,
    get_gemini_analysis_service,
    get_hybrid_service,
    get_feedback_collector,
    get_retrain_queue
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

            mediapipe_agreement = None
            if mp_features:
                gemini_shape = analysis_result.get("analysis", {}).get("face_shape")
                mediapipe_agreement = (
                    mp_features.face_shape in gemini_shape or
                    gemini_shape in mp_features.face_shape
                )

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
    face_detector: 'FaceDetectionService' = Depends(get_face_detection_service),
    gemini_service: 'GeminiAnalysisService' = Depends(get_gemini_analysis_service)
):
    """Face analysis and hairstyle recommendation (v20.2.0: ML integrated)"""
    start_time = time.time()
    image_hash = None

    try:
        if not settings.GEMINI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="Gemini API 키가 설정되지 않았습니다."
            )

        if not file.filename:
            raise HTTPException(status_code=400, detail="파일명이 없습니다")

        file_ext = file.filename.lower().split('.')[-1]
        if file_ext not in ['jpg', 'jpeg', 'png', 'webp']:
            raise InvalidFileFormatException()

        logger.info(f"이미지 업로드 시작: {file.filename}")

        image_data = await file.read()
        image_hash = calculate_image_hash(image_data)

        log_structured("analysis_start", {
            "filename": file.filename,
            "file_size_kb": round(len(image_data) / 1024, 2),
            "image_hash": image_hash[:16]
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
                "model_used": settings.MODEL_NAME
            }

        # Face detection using injected service
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

        # Gemini analysis using injected service (with MediaPipe hints)
        gemini_start = time.time()
        analysis_result = gemini_service.analyze_with_gemini(image_data, mp_features)
        gemini_time = round((time.time() - gemini_start) * 1000, 2)

        # Use MediaPipe results (for consistency)
        gender = None  # 성별 정보 추출 (NEW)
        if mp_features:
            face_shape = mp_features.face_shape
            skin_tone = mp_features.skin_tone
            gender = mp_features.gender  # 성별 추론 결과
            logger.info(f"✅ MediaPipe 결과 채택: {face_shape} / {skin_tone} / 성별: {gender} (일관성 보장)")

            # Log Gemini results for comparison
            gemini_face_shape = analysis_result.get("analysis", {}).get("face_shape")
            if gemini_face_shape != face_shape:
                logger.warning(f"⚠️ Gemini 불일치: {gemini_face_shape} (MediaPipe: {face_shape})")

            # Update analysis result with MediaPipe values
            analysis_result["analysis"]["face_shape"] = face_shape
            analysis_result["analysis"]["personal_color"] = skin_tone
            analysis_result["analysis"]["gender"] = gender  # 성별 정보 추가
        else:
            # Use Gemini results if MediaPipe failed
            face_shape = analysis_result.get("analysis", {}).get("face_shape")
            skin_tone = analysis_result.get("analysis", {}).get("personal_color")
            logger.warning(f"⚠️ MediaPipe 없음, Gemini 결과 사용: {face_shape} / {skin_tone}")

        # Add ML predictions and embeddings
        for idx, recommendation in enumerate(analysis_result.get("recommendations", []), 1):
            style_name = recommendation.get("style_name", "")

            # ✅ hairstyle_id 찾기 (ML 모델에서)
            from core.ml_loader import ml_recommender
            hairstyle_id = None
            if ml_recommender is not None:
                from utils.style_preprocessor import normalize_style_name
                normalized_name = normalize_style_name(style_name)
                hairstyle_id = ml_recommender.style_to_idx.get(normalized_name)
            recommendation['hairstyle_id'] = hairstyle_id

            # ML confidence score
            ml_score = predict_ml_score(face_shape, skin_tone, style_name)
            recommendation['ml_confidence'] = ml_score
            recommendation['confidence_level'] = get_confidence_level(ml_score)

            # ✅ score 필드 추가 (안드로이드 앱 호환성)
            recommendation['score'] = round(ml_score / 100.0, 2)  # 0-1 범위로 변환

            # Style embedding (Sentence Transformer)
            if sentence_transformer is not None:
                try:
                    embedding = sentence_transformer.encode(style_name)
                    recommendation['style_embedding'] = embedding.tolist()
                    logger.info(f"✅ 임베딩 생성 성공: {style_name} → {len(embedding)}차원")
                except Exception as e:
                    logger.error(f"❌ 임베딩 생성 실패 ({style_name}): {str(e)}")
                    recommendation['style_embedding'] = None
            else:
                recommendation['style_embedding'] = None

            # Naver search URL
            encoded_query = urllib.parse.quote(f"{style_name} 헤어스타일")
            recommendation["image_search_url"] = f"https://search.naver.com/search.naver?where=image&query={encoded_query}"

        # Cache result
        save_to_cache(image_hash, analysis_result)

        # Save to database
        total_time = round(time.time() - start_time, 2)
        analysis_id = save_to_database(
            image_hash=image_hash,
            analysis_result=analysis_result,
            processing_time=total_time,
            detection_method=face_result.get("method", "mediapipe"),
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
                "personal_color": skin_tone,
                "warning": "feedback_disabled"
            })

        log_structured("analysis_complete", {
            "image_hash": image_hash[:16],
            "processing_time": total_time,
            "face_detection_time_ms": face_detection_time,
            "gemini_analysis_time_ms": gemini_time,
            "mediapipe_enabled": mp_features is not None,
            "face_shape": face_shape,
            "personal_color": skin_tone,
            "analysis_id": analysis_id,
            "database_saved": analysis_id is not None
        })

        return {
            "success": True,
            "data": analysis_result,
            "analysis_id": analysis_id,
            "processing_time": total_time,
            "performance": {
                "face_detection_ms": face_detection_time,
                "gemini_analysis_ms": gemini_time,
                "detection_method": face_result.get("method", "mediapipe"),
                "mediapipe_analysis": "enabled" if mp_features else "failed"
            },
            "cached": False,
            "model_used": settings.MODEL_NAME,
            "feedback_enabled": analysis_id is not None  # ✅ NEW: Android can check this
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


@router.post("/v2/feedback")
@limiter.limit("20/minute")  # 분당 20회 제한 (피드백은 더 자주 사용)
async def collect_feedback(
    request: Request,
    face_shape: str,
    skin_tone: str,
    hairstyle_id: int,
    user_reaction: str,
    ml_prediction: float,
    user_id: str = "anonymous",
    collector: 'FeedbackCollector' = Depends(get_feedback_collector),
    retrain_q: 'RetrainQueue' = Depends(get_retrain_queue)
):
    """
    User feedback collection endpoint (v2)

    Args:
        face_shape: Face shape ("계란형", "둥근형", "긴형", "각진형")
        skin_tone: Skin tone ("가을웜", "겨울쿨", "봄웜", "여름쿨")
        hairstyle_id: Hairstyle ID (0-based index)
        user_reaction: "👍" (like) or "👎" (dislike)
        ml_prediction: ML model prediction score
        user_id: User ID (default: "anonymous")

    Returns:
        {"total_feedbacks": int, "retrain_triggered": bool, "retrain_job_id": str}

    Ground Truth Rules:
        👍 -> 90.0 (user LIKED this combination)
        👎 -> 10.0 (user DISLIKED this combination)
    """
    try:
        # Input validation
        if user_reaction not in ["👍", "👎"]:
            raise HTTPException(
                status_code=400,
                detail="user_reaction은 '👍' 또는 '👎'만 가능합니다."
            )

        # Save feedback using injected collector
        result = collector.save_feedback(
            face_shape=face_shape,
            skin_tone=skin_tone,
            hairstyle_id=hairstyle_id,
            user_reaction=user_reaction,
            ml_prediction=ml_prediction,
            user_id=user_id
        )

        retrain_job_id = None

        # Check for retrain trigger using injected queue
        if result['retrain_triggered']:
            job = retrain_q.add_job(result['total_feedbacks'])
            retrain_job_id = job['job_id']

            logger.info(
                f"🔄 재학습 작업 생성: {retrain_job_id} "
                f"(피드백 {result['total_feedbacks']}개)"
            )

            log_structured("retrain_job_created", {
                "job_id": retrain_job_id,
                "feedback_count": result['total_feedbacks']
            })

        logger.info(
            f"✅ 피드백 수집 완료: {face_shape} + {skin_tone} + ID#{hairstyle_id} "
            f"-> {user_reaction} | Total: {result['total_feedbacks']}"
        )

        log_structured("feedback_collected", {
            "face_shape": face_shape,
            "skin_tone": skin_tone,
            "hairstyle_id": hairstyle_id,
            "user_reaction": user_reaction,
            "ml_prediction": ml_prediction,
            "total_feedbacks": result['total_feedbacks'],
            "retrain_triggered": result['retrain_triggered'],
            "retrain_job_id": retrain_job_id
        })

        return {
            "success": True,
            "total_feedbacks": result['total_feedbacks'],
            "retrain_triggered": result['retrain_triggered'],
            "retrain_job_id": retrain_job_id,
            "message": "피드백이 성공적으로 저장되었습니다."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 피드백 수집 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"피드백 수집 중 오류가 발생했습니다: {str(e)}"
        )
