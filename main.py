"""
HairMe Backend - AI-powered Hairstyle Recommendation Service
Version: 20.2.0 (MediaPipe transition complete)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from core.logging import logger, log_structured
from core.cache import init_redis
from core.ml_loader import load_ml_model, load_sentence_transformer
from database.connection import init_database
from database.migration import migrate_database_schema
from models.mediapipe_analyzer import MediaPipeFaceAnalyzer
from services.hybrid_recommender import get_hybrid_service
from services.feedback_collector import get_feedback_collector
from services.retrain_queue import get_retrain_queue
from routers.admin import router as admin_router
from api.endpoints.analyze import router as analyze_router, init_gemini
from api.endpoints.feedback import router as feedback_router
import api.endpoints.analyze as analyze_module


# ========== FastAPI App Initialization ==========
app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION
)


# ========== CORS Middleware ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== Register Routers ==========
app.include_router(admin_router, prefix="/api", tags=["admin"])
app.include_router(analyze_router, prefix="/api", tags=["analysis"])
app.include_router(feedback_router, prefix="/api", tags=["feedback"])


# ========== Startup Event ==========
@app.on_event("startup")
async def startup_event():
    """Initialize all services on server startup"""
    logger.info("🚀 서버 시작 중...")

    # Initialize Gemini API
    init_gemini()

    # Initialize MediaPipe Face Analyzer
    try:
        analyzer = MediaPipeFaceAnalyzer()
        analyze_module.mediapipe_analyzer = analyzer
        logger.info("✅ MediaPipe 얼굴 분석기 초기화 완료")
        log_structured("mediapipe_initialized", {
            "status": "success",
            "landmarks": 478
        })
    except Exception as e:
        logger.error(f"❌ MediaPipe 초기화 실패: {str(e)}")
        analyze_module.mediapipe_analyzer = None
        log_structured("mediapipe_initialized", {
            "status": "failed",
            "error": str(e)
        })

    # Load ML Model
    ml_loaded = load_ml_model()
    if ml_loaded:
        logger.info("✅ ML 모드: 활성화")
        log_structured("ml_model_loaded", {
            "status": "success",
            "model_path": settings.ML_MODEL_PATH
        })
    else:
        logger.warning("⚠️ ML 모드: 비활성화 (기본 점수 사용)")
        log_structured("ml_model_loaded", {
            "status": "failed",
            "fallback": "default_score"
        })

    # Load Sentence Transformer
    st_loaded = load_sentence_transformer()
    if st_loaded:
        logger.info("✅ 스타일 임베딩: 활성화")
        log_structured("sentence_transformer_loaded", {
            "status": "success",
            "model": settings.SENTENCE_TRANSFORMER_MODEL,
            "embedding_dim": 384
        })
    else:
        logger.warning("⚠️ 스타일 임베딩: 비활성화 (임베딩 없이 진행)")
        log_structured("sentence_transformer_loaded", {
            "status": "failed",
            "fallback": "no_embedding"
        })

    # Initialize Hybrid Recommendation Service (Gemini + ML)
    try:
        if settings.GEMINI_API_KEY:
            service = get_hybrid_service(settings.GEMINI_API_KEY)
            analyze_module.hybrid_service = service
            logger.info("✅ 하이브리드 추천 서비스 초기화 완료 (Gemini + ML)")
            log_structured("hybrid_service_initialized", {
                "status": "success",
                "gemini_model": settings.MODEL_NAME,
                "ml_model": "hairstyle_recommender.pt"
            })
        else:
            logger.warning("⚠️ GEMINI_API_KEY 없음 - 하이브리드 서비스 비활성화")
            analyze_module.hybrid_service = None
    except Exception as e:
        logger.error(f"❌ 하이브리드 서비스 초기화 실패: {str(e)}")
        analyze_module.hybrid_service = None
        log_structured("hybrid_service_initialized", {
            "status": "failed",
            "error": str(e)
        })

    # Initialize Feedback Collector
    try:
        collector = get_feedback_collector()
        analyze_module.feedback_collector = collector
        logger.info("✅ 피드백 수집기 초기화 완료")
        log_structured("feedback_collector_initialized", {
            "status": "success"
        })
    except Exception as e:
        logger.error(f"❌ 피드백 수집기 초기화 실패: {str(e)}")
        analyze_module.feedback_collector = None
        log_structured("feedback_collector_initialized", {
            "status": "failed",
            "error": str(e)
        })

    # Initialize Retrain Queue
    try:
        queue = get_retrain_queue()
        analyze_module.retrain_queue = queue
        logger.info("✅ 재학습 큐 초기화 완료")
        log_structured("retrain_queue_initialized", {
            "status": "success"
        })
    except Exception as e:
        logger.error(f"❌ 재학습 큐 초기화 실패: {str(e)}")
        analyze_module.retrain_queue = None
        log_structured("retrain_queue_initialized", {
            "status": "failed",
            "error": str(e)
        })

    # Initialize Database
    db_initialized = init_database()
    if db_initialized:
        # Run schema migration
        migrate_database_schema()

    # Initialize Redis Cache
    init_redis()


# ========== Root Endpoint ==========
@app.get("/")
async def root():
    """Root endpoint with service status"""
    return {
        "message": f"{settings.APP_TITLE} - v{settings.APP_VERSION} (MediaPipe 전환 완료)",
        "version": settings.APP_VERSION,
        "model": settings.MODEL_NAME,
        "status": "running",
        "features": {
            "mediapipe_analysis": "enabled" if analyze_module.mediapipe_analyzer else "disabled",
            "gemini_analysis": "enabled" if settings.GEMINI_API_KEY else "disabled",
            "redis_cache": "enabled",
            "database": "enabled",
            "feedback_system": "enabled",
            "ml_prediction": "enabled",
            "style_embedding": "enabled"
        }
    }


# ========== Health Check Endpoint ==========
@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "model": settings.MODEL_NAME,
        "mediapipe_analysis": "enabled" if analyze_module.mediapipe_analyzer else "disabled",
        "gemini_api": "configured" if settings.GEMINI_API_KEY else "not_configured",
        "redis": "connected",
        "database": "connected",
        "feedback_system": "enabled",
        "ml_model": "enabled",
        "style_embedding": "enabled"
    }


# ========== Lambda Handler ==========
# For AWS Lambda deployment using Mangum
try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
    logger.info("✅ Lambda handler initialized")
except ImportError:
    logger.warning("⚠️ Mangum not installed - Lambda handler not available")
    handler = None


# ========== Main Entry Point ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
