"""HairMe Chatbot Service - 독립 실행 진입점

기존 헤어스타일/뷰티 서버(main.py)와 분리된 별도 FastAPI 앱.
- 챗봇만 단일 책임으로 띄움 → 배포/스케일링/장애 격리가 깔끔
- LangGraph 워크플로우 + SSE 스트리밍만 마운트

실행:
    uvicorn services.chatbot.main:app --port 8002 --reload

또는:
    python -m services.chatbot.main
"""
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config.settings import settings
from core.logging import logger
from api.endpoints.chatbot import router as chatbot_router


# ========== FastAPI App ==========
app = FastAPI(
    title="HairMe Chatbot API",
    description="LangGraph 기반 헤어스타일 RAG 챗봇 (독립 서비스)",
    version="0.1.0",
)


# ========== Rate Limiter ==========
limiter = Limiter(key_func=get_remote_address, config_filename="__no_env__")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ========== CORS ==========
# 챗봇은 SSE를 쓰므로 GET/POST/OPTIONS만 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)


# ========== Security Headers ==========
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if "Server" in response.headers:
        del response.headers["Server"]
    return response


# ========== Router ==========
app.include_router(chatbot_router, prefix="/api", tags=["chatbot"])


# ========== Startup ==========
@app.on_event("startup")
async def startup_event():
    """필수 환경 변수 검증 + 그래프 사전 컴파일"""
    logger.info("HairMe Chatbot service starting...")

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set — embeddings/LLM will fail")
        raise RuntimeError("OPENAI_API_KEY environment variable is required")

    if not os.getenv("TAVILY_API_KEY"):
        logger.warning(
            "TAVILY_API_KEY not set — web_search 노드는 fallback 메시지만 반환합니다"
        )

    # 첫 요청 지연을 줄이기 위해 그래프를 startup에서 미리 컴파일
    from api.endpoints.chatbot import _get_graph

    _get_graph()
    logger.info("LangGraph workflow compiled and ready")


# ========== Root / Health ==========
@app.get("/")
async def root():
    return {
        "name": "HairMe Chatbot API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "chat": "/api/chatbot/chat",
            "stream": "/api/chatbot/chat/stream",
        },
    }


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "chatbot"}


# ========== Entry Point ==========
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
