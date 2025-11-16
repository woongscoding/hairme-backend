"""Face analysis and hairstyle recommendation endpoints"""

import os
import time
import json
import io
import urllib.parse
from typing import Optional, Dict, Any, Union
from datetime import datetime

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import google.generativeai as genai

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
from database.models import AnalysisHistory
from database.connection import get_db_session
from models.mediapipe_analyzer import MediaPipeFaceFeatures


router = APIRouter()


# Global variables (initialized in main.py startup)
mediapipe_analyzer = None
hybrid_service = None
feedback_collector = None
retrain_queue = None


# ========== Gemini Configuration ==========
ANALYSIS_PROMPT = """분석하고 JSON으로 응답:

얼굴형: 계란형/둥근형/각진형/긴형 중 1개
퍼스널컬러: 봄웜/가을웜/여름쿨/겨울쿨 중 1개
헤어스타일 추천 3개 (각 이름 15자, 이유 30자 이내)

JSON 형식:
{
  "analysis": {
    "face_shape": "계란형",
    "personal_color": "봄웜",
    "features": "이목구비 특징"
  },
  "recommendations": [
    {"style_name": "스타일명", "reason": "추천 이유"}
  ]
}"""


def init_gemini() -> None:
    """Initialize Gemini API"""
    if not settings.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY 환경변수가 설정되지 않았습니다!")
    else:
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            logger.info("✅ Gemini API 초기화 완료")
        except Exception as e:
            logger.error(f"Gemini API 초기화 실패: {str(e)}")


# ========== Helper Functions ==========
def verify_face_with_gemini(image_data: bytes) -> Dict[str, Any]:
    """
    Verify face with Gemini when OpenCV fails

    Args:
        image_data: Image binary data

    Returns:
        Dictionary with face verification results
    """
    try:
        image = Image.open(io.BytesIO(image_data))
        image.thumbnail((256, 256))

        model = genai.GenerativeModel(settings.MODEL_NAME)
        prompt = """이미지에 사람 얼굴이 있나요?

JSON으로만 답변:
{"has_face": true/false, "face_count": 숫자}"""

        response = model.generate_content([prompt, image])
        result = json.loads(response.text.strip())

        return {
            "has_face": result.get("has_face", False),
            "face_count": result.get("face_count", 0),
            "method": "gemini"
        }

    except Exception as e:
        logger.error(f"Gemini 얼굴 검증 실패: {str(e)}")
        return {
            "has_face": False,
            "face_count": 0,
            "method": "gemini",
            "error": str(e)
        }


def detect_face(image_data: bytes) -> Dict[str, Any]:
    """
    Detect face (MediaPipe first, fallback to Gemini)

    Args:
        image_data: Image binary data

    Returns:
        Dictionary with face detection results
    """
    # 1st attempt: MediaPipe (most accurate - 90%+)
    if mediapipe_analyzer is not None:
        try:
            mp_features = mediapipe_analyzer.analyze(image_data)

            if mp_features:
                log_structured("face_detection", {
                    "method": "mediapipe",
                    "face_count": 1,
                    "success": True,
                    "face_shape": mp_features.face_shape,
                    "skin_tone": mp_features.skin_tone,
                    "confidence": mp_features.confidence
                })
                return {
                    "has_face": True,
                    "face_count": 1,
                    "method": "mediapipe",
                    "features": mp_features
                }

        except Exception as e:
            logger.warning(f"MediaPipe 얼굴 감지 실패: {str(e)}")

    # 2nd attempt: Gemini (final fallback)
    logger.info("MediaPipe 실패, Gemini로 얼굴 검증 시작...")
    gemini_result = verify_face_with_gemini(image_data)

    log_structured("face_detection", {
        "method": "gemini",
        "face_count": gemini_result.get("face_count", 0),
        "success": gemini_result.get("has_face", False)
    })

    return gemini_result


def analyze_with_gemini(
    image_data: bytes,
    mp_features: Optional[MediaPipeFaceFeatures] = None
) -> Dict[str, Any]:
    """
    Analyze face with Gemini Vision API (with MediaPipe hints)

    Args:
        image_data: Image binary data
        mp_features: MediaPipe analysis results (optional)

    Returns:
        Dictionary with analysis results
    """
    try:
        image = Image.open(io.BytesIO(image_data))

        # Provide MediaPipe hints if available
        if mp_features:
            prompt = f"""다음 얼굴 사진을 분석하고 JSON으로 응답해주세요.

🔍 **MediaPipe 측정 데이터** (수학적 얼굴 분석 - 신뢰도 {mp_features.confidence:.0%}):
- 얼굴형: {mp_features.face_shape}
- 피부톤: {mp_features.skin_tone}
- 얼굴 비율(높이/너비): {mp_features.face_ratio:.2f}
- 이마 너비: {mp_features.forehead_width:.0f}px
- 광대 너비: {mp_features.cheekbone_width:.0f}px
- 턱 너비: {mp_features.jaw_width:.0f}px
- ITA 값: {mp_features.ITA_value:.1f}°

⚠️ **중요**: 위 MediaPipe 측정값은 수학적으로 계산된 정확한 데이터입니다.
시각적으로 명백히 다르지 않다면 MediaPipe 결과를 그대로 사용하세요.
(참고: 최종 결과는 MediaPipe 값이 우선 채택되므로, 일관성을 위해 같은 값 사용 권장)

**분석 항목:**
1. 얼굴형: 계란형/둥근형/각진형/긴형/하트형 중 1개
2. 퍼스널컬러: 봄웜/가을웜/여름쿨/겨울쿨 중 1개
3. 헤어스타일 추천 3개 (각 이름 15자, 이유 30자 이내)

**JSON 형식:**
{{
  "analysis": {{
    "face_shape": "계란형",
    "personal_color": "봄웜",
    "features": "이목구비 특징 설명"
  }},
  "recommendations": [
    {{"style_name": "스타일명", "reason": "추천 이유"}}
  ]
}}"""
            logger.info(f"✅ MediaPipe 힌트 적용: {mp_features.face_shape} / {mp_features.skin_tone}")

        else:
            # Default prompt without MediaPipe hints
            prompt = ANALYSIS_PROMPT
            logger.warning("⚠️ MediaPipe 특징 없음, 기본 프롬프트 사용")

        model = genai.GenerativeModel(settings.MODEL_NAME)

        # Use temperature=0 for consistent responses
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,
        )

        response = model.generate_content(
            [prompt, image],
            generation_config=generation_config
        )

        raw_text = response.text.strip()

        # Clean up markdown code blocks
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

        result = json.loads(raw_text.strip())

        logger.info(f"✅ Gemini 분석 성공: {result.get('analysis', {}).get('face_shape')}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 실패: {str(e)}\n응답 내용: {response.text[:200]}")
        raise HTTPException(
            status_code=500,
            detail=f"AI 응답 파싱 실패: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Gemini 분석 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"AI 분석 중 오류가 발생했습니다: {str(e)}"
        )


def save_to_database(
    image_hash: str,
    analysis_result: Dict[str, Any],
    processing_time: float,
    detection_method: str,
    mp_features: Optional[MediaPipeFaceFeatures] = None
) -> Optional[Union[int, str]]:
    """
    Save analysis result to database (MySQL or DynamoDB)

    Supports both MySQL (RDS) and DynamoDB backends based on USE_DYNAMODB env variable.

    Args:
        image_hash: SHA256 hash of the image
        analysis_result: Analysis result dictionary
        processing_time: Processing time in seconds
        detection_method: Detection method used
        mp_features: MediaPipe features (optional)

    Returns:
        Record ID if successful (int for MySQL, str for DynamoDB), None otherwise
    """
    use_dynamodb = os.getenv('USE_DYNAMODB', 'false').lower() == 'true'

    # ========== DynamoDB Backend ==========
    if use_dynamodb:
        try:
            from database.dynamodb_connection import save_analysis

            gemini_shape = analysis_result.get("analysis", {}).get("face_shape")
            recommendations = analysis_result.get("recommendations", [])

            # Calculate MediaPipe agreement
            mediapipe_agreement = None
            if mp_features:
                mediapipe_agreement = (
                    mp_features.face_shape in gemini_shape or
                    gemini_shape in mp_features.face_shape
                )

            # Build data dict for DynamoDB
            data = {
                'image_hash': image_hash,
                'face_shape': gemini_shape,
                'personal_color': analysis_result.get("analysis", {}).get("personal_color"),
                'recommendations': recommendations,
                'recommended_styles': recommendations,
                'processing_time': processing_time,
                'detection_method': detection_method,
                'opencv_gemini_agreement': mediapipe_agreement,
            }

            # Add MediaPipe continuous features
            if mp_features:
                data['mediapipe_face_ratio'] = mp_features.face_ratio
                data['mediapipe_forehead_width'] = mp_features.forehead_width
                data['mediapipe_cheekbone_width'] = mp_features.cheekbone_width
                data['mediapipe_jaw_width'] = mp_features.jaw_width

                # Ratios (division by zero protection)
                if mp_features.cheekbone_width > 0:
                    data['mediapipe_forehead_ratio'] = mp_features.forehead_width / mp_features.cheekbone_width
                    data['mediapipe_jaw_ratio'] = mp_features.jaw_width / mp_features.cheekbone_width

                # Skin measurements
                data['mediapipe_ITA_value'] = mp_features.ITA_value
                data['mediapipe_hue_value'] = mp_features.hue_value

                # Metadata
                data['mediapipe_confidence'] = mp_features.confidence
                data['mediapipe_features_complete'] = True

                logger.info(f"✅ MediaPipe 연속형 변수 포함: ratio={mp_features.face_ratio:.2f}, ITA={mp_features.ITA_value:.1f}")

            # Save to DynamoDB
            analysis_id = save_analysis(data)

            if analysis_id:
                logger.info(f"✅ DynamoDB 저장 성공 (ID: {analysis_id})")
                log_structured("database_saved", {
                    "backend": "dynamodb",
                    "analysis_id": analysis_id,
                    "mediapipe_enabled": mp_features is not None,
                    "mediapipe_agreement": mediapipe_agreement,
                    "recommendations_count": len(recommendations)
                })
                return analysis_id
            else:
                logger.error("❌ DynamoDB 저장 실패")
                return None

        except Exception as e:
            logger.error(f"❌ DynamoDB 저장 실패: {str(e)}")
            return None

    # ========== MySQL Backend (Original) ==========
    else:
        db = get_db_session()
        if not db:
            logger.warning("⚠️ 데이터베이스 연결이 없어 저장을 생략합니다.")
            return None

        try:
            gemini_shape = analysis_result.get("analysis", {}).get("face_shape")

            # Calculate MediaPipe agreement
            mediapipe_agreement = None
            if mp_features:
                mediapipe_agreement = (
                    mp_features.face_shape in gemini_shape or
                    gemini_shape in mp_features.face_shape
                )

            recommendations = analysis_result.get("recommendations", [])

            history = AnalysisHistory(
                image_hash=image_hash,
                face_shape=gemini_shape,
                personal_color=analysis_result.get("analysis", {}).get("personal_color"),
                recommendations=recommendations,
                recommended_styles=recommendations,
                processing_time=processing_time,
                detection_method=detection_method,
                opencv_gemini_agreement=mediapipe_agreement,
            )

            # ✅ MediaPipe 연속형 변수 저장
            if mp_features:
                # 얼굴 측정값
                history.mediapipe_face_ratio = mp_features.face_ratio
                history.mediapipe_forehead_width = mp_features.forehead_width
                history.mediapipe_cheekbone_width = mp_features.cheekbone_width
                history.mediapipe_jaw_width = mp_features.jaw_width

                # 비율 계산 (division by zero 방지)
                if mp_features.cheekbone_width > 0:
                    history.mediapipe_forehead_ratio = mp_features.forehead_width / mp_features.cheekbone_width
                    history.mediapipe_jaw_ratio = mp_features.jaw_width / mp_features.cheekbone_width

                # 피부 측정값
                history.mediapipe_ITA_value = mp_features.ITA_value
                history.mediapipe_hue_value = mp_features.hue_value

                # 메타데이터
                history.mediapipe_confidence = mp_features.confidence
                history.mediapipe_features_complete = True

                logger.info(f"✅ MediaPipe 연속형 변수 저장: ratio={mp_features.face_ratio:.2f}, ITA={mp_features.ITA_value:.1f}")

            db.add(history)
            db.commit()
            db.refresh(history)

            logger.info(f"✅ MySQL 저장 성공 (ID: {history.id})")
            log_structured("database_saved", {
                "backend": "mysql",
                "record_id": history.id,
                "mediapipe_enabled": mp_features is not None,
                "mediapipe_agreement": mediapipe_agreement,
                "recommendations_count": len(recommendations)
            })

            db.close()
            return history.id

        except Exception as e:
            logger.error(f"❌ MySQL 저장 실패: {str(e)}")
            db.close()
            return None


# ========== API Endpoints ==========
@router.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
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

        # Face detection
        face_detection_start = time.time()
        face_result = detect_face(image_data)
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

        # Gemini analysis (with MediaPipe hints)
        gemini_start = time.time()
        analysis_result = analyze_with_gemini(image_data, mp_features)
        gemini_time = round((time.time() - gemini_start) * 1000, 2)

        # Use MediaPipe results (for consistency)
        if mp_features:
            face_shape = mp_features.face_shape
            skin_tone = mp_features.skin_tone
            logger.info(f"✅ MediaPipe 결과 채택: {face_shape} / {skin_tone} (일관성 보장)")

            # Log Gemini results for comparison
            gemini_face_shape = analysis_result.get("analysis", {}).get("face_shape")
            if gemini_face_shape != face_shape:
                logger.warning(f"⚠️ Gemini 불일치: {gemini_face_shape} (MediaPipe: {face_shape})")

            # Update analysis result with MediaPipe values
            analysis_result["analysis"]["face_shape"] = face_shape
            analysis_result["analysis"]["personal_color"] = skin_tone
        else:
            # Use Gemini results if MediaPipe failed
            face_shape = analysis_result.get("analysis", {}).get("face_shape")
            skin_tone = analysis_result.get("analysis", {}).get("personal_color")
            logger.warning(f"⚠️ MediaPipe 없음, Gemini 결과 사용: {face_shape} / {skin_tone}")

        # Add ML predictions and embeddings
        for idx, recommendation in enumerate(analysis_result.get("recommendations", []), 1):
            style_name = recommendation.get("style_name", "")

            # ML confidence score
            ml_score = predict_ml_score(face_shape, skin_tone, style_name)
            recommendation['ml_confidence'] = ml_score
            recommendation['confidence_level'] = get_confidence_level(ml_score)

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

        log_structured("analysis_complete", {
            "image_hash": image_hash[:16],
            "processing_time": total_time,
            "face_detection_time_ms": face_detection_time,
            "gemini_analysis_time_ms": gemini_time,
            "mediapipe_enabled": mp_features is not None,
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
                "gemini_analysis_ms": gemini_time,
                "detection_method": face_result.get("method", "mediapipe"),
                "mediapipe_analysis": "enabled" if mp_features else "failed"
            },
            "cached": False,
            "model_used": settings.MODEL_NAME
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
        logger.error(f"분석 중 오류 발생: {str(e)}")

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
                "message": f"분석 중 오류가 발생했습니다: {str(e)}"
            }
        )


@router.post("/v2/analyze-hybrid")
async def analyze_face_hybrid(file: UploadFile = File(...)):
    """
    Hybrid face analysis and hairstyle recommendation (Gemini + ML)

    Flow:
    1. Analyze face shape + skin tone with MediaPipe
    2. Get 4 recommendations from Gemini API
    3. Get Top-3 recommendations from ML model
    4. Return up to 7 recommendations after deduplication
    """
    start_time = time.time()

    try:
        # File validation
        if not file.filename:
            raise HTTPException(status_code=400, detail="파일명이 없습니다")

        file_ext = file.filename.lower().split('.')[-1]
        if file_ext not in ['jpg', 'jpeg', 'png', 'webp']:
            raise InvalidFileFormatException()

        logger.info(f"🎨 하이브리드 분석 시작: {file.filename}")

        # Read image
        image_data = await file.read()
        image_hash = calculate_image_hash(image_data)

        # 1. MediaPipe face analysis
        if not mediapipe_analyzer:
            raise HTTPException(
                status_code=500,
                detail="MediaPipe 분석기가 초기화되지 않았습니다."
            )

        mp_features = mediapipe_analyzer.analyze(image_data)

        if not mp_features:
            raise NoFaceDetectedException()

        face_shape = mp_features.face_shape
        skin_tone = mp_features.skin_tone

        logger.info(f"✅ MediaPipe 분석: {face_shape} + {skin_tone}")

        # 2. Hybrid recommendation
        if not hybrid_service:
            raise HTTPException(
                status_code=500,
                detail="하이브리드 추천 서비스가 초기화되지 않았습니다."
            )

        recommendation_result = hybrid_service.recommend(
            image_data, face_shape, skin_tone
        )

        # 3. Add Naver search URLs
        for rec in recommendation_result.get("recommendations", []):
            style_name = rec.get("style_name", "")
            encoded_query = urllib.parse.quote(f"{style_name} 헤어스타일")
            rec["image_search_url"] = f"https://search.naver.com/search.naver?where=image&query={encoded_query}"

        # 4. Save to database
        total_time = round(time.time() - start_time, 2)
        analysis_id = None

        use_dynamodb = os.getenv('USE_DYNAMODB', 'false').lower() == 'true'

        if use_dynamodb:
            # Save to DynamoDB
            try:
                from database.dynamodb_connection import save_analysis

                data = {
                    'user_id': 'anonymous',
                    'image_hash': image_hash,
                    'face_shape': face_shape,
                    'personal_color': skin_tone,
                    'recommendations': recommendation_result.get("recommendations", []),
                    'recommended_styles': recommendation_result.get("recommendations", []),
                    'processing_time': total_time,
                    'detection_method': 'hybrid',
                }

                analysis_id = save_analysis(data)
                logger.info(f"✅ DynamoDB 저장 완료: analysis_id={analysis_id}")

            except Exception as e:
                logger.error(f"❌ DynamoDB 저장 실패: {str(e)}")
        else:
            # Save to MySQL
            db = get_db_session()
            if db:
                try:
                    new_record = AnalysisHistory(
                        user_id="anonymous",
                        image_hash=image_hash,
                        face_shape=face_shape,
                        personal_color=skin_tone,
                        recommendations=recommendation_result,
                        processing_time=total_time,
                        detection_method="hybrid",
                        recommended_styles=recommendation_result.get("recommendations", [])
                    )

                    db.add(new_record)
                    db.commit()
                    db.refresh(new_record)

                    analysis_id = new_record.id

                    logger.info(f"✅ MySQL 저장 완료: analysis_id={analysis_id}")

                    db.close()
                except Exception as e:
                    logger.error(f"❌ MySQL 저장 실패: {str(e)}")
                    db.close()

        logger.info(f"✅ 하이브리드 분석 완료 ({total_time}초)")

        return {
            "success": True,
            "analysis_id": analysis_id,
            "data": recommendation_result,
            "processing_time": total_time,
            "method": "hybrid",
            "mediapipe_features": {
                "face_shape": face_shape,
                "skin_tone": skin_tone,
                "confidence": mp_features.confidence
            },
            "model_used": "gemini-1.5-flash-latest + hairstyle_recommender.pt"
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
        logger.error(f"❌ 하이브리드 분석 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"분석 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/v2/feedback")
async def collect_feedback(
    face_shape: str,
    skin_tone: str,
    hairstyle_id: int,
    user_reaction: str,
    ml_prediction: float,
    user_id: str = "anonymous"
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
    if not feedback_collector:
        raise HTTPException(
            status_code=500,
            detail="피드백 수집기가 초기화되지 않았습니다."
        )

    try:
        # Input validation
        if user_reaction not in ["👍", "👎"]:
            raise HTTPException(
                status_code=400,
                detail="user_reaction은 '👍' 또는 '👎'만 가능합니다."
            )

        # Save feedback
        result = feedback_collector.save_feedback(
            face_shape=face_shape,
            skin_tone=skin_tone,
            hairstyle_id=hairstyle_id,
            user_reaction=user_reaction,
            ml_prediction=ml_prediction,
            user_id=user_id
        )

        retrain_job_id = None

        # Check for retrain trigger
        if result['retrain_triggered'] and retrain_queue:
            job = retrain_queue.add_job(result['total_feedbacks'])
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
