"""HairMe LangGraph 챗봇 엔드포인트

두 가지 응답 모드:
- POST /chatbot/chat       : 동기 (한 번에 답변 반환)
- POST /chatbot/chat/stream: SSE 스트리밍 (노드별/토큰별 진행 상황 실시간)

설계 의도:
- 그래프 컴파일은 모듈 로드 시 1회 → 요청마다 재컴파일 비용 회피
- ainvoke / astream으로 비동기 처리 → FastAPI 이벤트 루프 차단 방지
- 응답 스키마를 Pydantic으로 명시 → 클라이언트(Flutter)에서 타입 안전
"""
import json
from typing import AsyncIterator, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.logging import logger
from services.chatbot.graph import create_graph

router = APIRouter()
# config_filename에 존재하지 않는 경로를 넘겨 slowapi/starlette가 .env를 읽지 않게 함
# (Korean Windows에서 cp949로 .env(UTF-8 한글)를 디코딩하다 UnicodeDecodeError가 나는 이슈 회피)
limiter = Limiter(key_func=get_remote_address, config_filename="__no_env__")

# 그래프는 모듈 로드 시 1회만 컴파일 (싱글톤)
# - 매 요청마다 컴파일하면 노드/엣지 재구성 비용이 들어감
# - 벡터DB도 retrieve 노드 내부에서 lazy 싱글톤으로 캐시됨
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        logger.info("Compiling LangGraph chatbot workflow...")
        _graph = create_graph()
    return _graph


# ---- 요청/응답 스키마 ----


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="사용자 질문")


class ChatResponse(BaseModel):
    question: str
    question_type: Optional[str] = Field(None, description="라우터가 분류한 카테고리")
    answer: str
    documents_used: int = Field(0, description="답변 생성에 사용된 문서 수")


# ---- 동기 엔드포인트 ----


@router.post("/chatbot/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """LangGraph 챗봇 — 동기 응답

    그래프 전체를 ainvoke로 실행하고 최종 State를 응답으로 변환.
    """
    try:
        graph = _get_graph()
        result = await graph.ainvoke({"question": body.question})

        return ChatResponse(
            question=body.question,
            question_type=result.get("question_type"),
            answer=result.get("generation", ""),
            documents_used=len(result.get("documents", []) or []),
        )
    except Exception:
        # 보안: 내부 예외 메시지 노출 금지 — 로그만 남기고 일반 메시지 반환
        logger.exception("Chatbot graph invocation failed")
        raise HTTPException(status_code=500, detail="챗봇 응답 생성에 실패했습니다.")


# ---- 스트리밍 엔드포인트 (Step 10) ----


async def _sse_event_stream(question: str) -> AsyncIterator[str]:
    """LangGraph astream을 SSE 형식으로 변환

    SSE 포맷: 'data: <json>\\n\\n'

    각 노드가 끝날 때마다 그 노드의 State 업데이트를 이벤트로 보냄.
    클라이언트는 이걸로 진행 상황(라우팅 결과, 검색 결과, 최종 답변)을
    실시간 표시할 수 있음.
    """
    graph = _get_graph()

    try:
        async for chunk in graph.astream({"question": question}):
            # chunk = {"노드이름": {업데이트된 State 일부}}
            for node_name, state_update in chunk.items():
                event = {
                    "node": node_name,
                    "update": _safe_serialize(state_update),
                }
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        # 종료 이벤트
        yield "data: [DONE]\n\n"
    except Exception:
        logger.exception("Chatbot stream failed")
        err = {"error": "스트림 생성 중 오류가 발생했습니다."}
        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"


def _safe_serialize(obj):
    """State 업데이트 dict를 SSE에 실어보낼 수 있는 형태로 정리"""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


@router.post("/chatbot/chat/stream")
@limiter.limit("30/minute")
async def chat_stream(request: Request, body: ChatRequest) -> StreamingResponse:
    """LangGraph 챗봇 — SSE 스트리밍

    응답 예 (한 줄씩 전송):
        data: {"node": "route_question", "update": {"question_type": "hairstyle"}}
        data: {"node": "retrieve", "update": {"documents": ["..."]}}
        data: {"node": "grade_documents", "update": {"web_search_needed": "no"}}
        data: {"node": "generate", "update": {"generation": "..."}}
        data: [DONE]

    Flutter 측에서 EventSource 또는 http stream으로 받아 단계별 UI 업데이트.
    """
    return StreamingResponse(
        _sse_event_stream(body.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # nginx 버퍼링 방지
        },
    )
