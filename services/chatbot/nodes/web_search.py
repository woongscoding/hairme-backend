"""
웹 검색 노드 (Web Search)

역할: 벡터DB에 답이 없을 때 Tavily로 실시간 웹 검색해서 documents에 추가

면접 포인트:
"내부 문서가 커버하지 못하는 질문(트렌드, 최신 정보)은 웹 검색으로 보완합니다.
HR 챗봇이라면 사내 규정으로 답할 수 없는 일반 노동법 질문 같은 케이스에
동일한 패턴을 적용할 수 있습니다."
"""
import os
from langchain_community.tools.tavily_search import TavilySearchResults

from services.chatbot.state import GraphState

_tavily = None


def _get_tavily():
    global _tavily
    if _tavily is None:
        # TAVILY_API_KEY 환경변수 필요
        _tavily = TavilySearchResults(max_results=3)
    return _tavily


def web_search(state: GraphState) -> dict:
    """Tavily 웹 검색

    입력: state["question"], state["documents"] (있으면 그대로 유지)
    출력: {"documents": 기존 + 웹 결과}
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", []) or []

    if not os.getenv("TAVILY_API_KEY"):
        # API 키 없을 때도 그래프가 죽지 않게 안내 메시지를 문서로 주입
        print("  [경고] TAVILY_API_KEY 미설정 — 빈 결과 반환")
        documents.append(
            "(웹 검색을 사용할 수 없습니다. TAVILY_API_KEY를 설정하세요.)"
        )
        return {"documents": documents}

    tavily = _get_tavily()
    results = tavily.invoke({"query": question})

    # results는 [{"url":..., "content":...}, ...] 형태
    for r in results:
        documents.append(r.get("content", ""))

    print(f"  웹 결과 {len(results)}건 추가")
    return {"documents": documents}
