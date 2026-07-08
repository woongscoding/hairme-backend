"""
HairMe 챗봇 그래프 정의

전체 흐름:
                    [route_question] (Step 8)
                          │
              ┌───────────┼───────────┐
              │ hairstyle │ app_usage │ general
              ▼           ▼           │
          [retrieve] (Step 4)         │
              │                       │
              ▼                       │
       [grade_documents] (Step 5)     │
              │                       │
       ┌──────┴──────┐                │
       │ web_search? │                │
       │   "yes"     │   "no"         │
       ▼             │                ▼
   [web_search]──────┼──────► [web_search] ──┐
       │             │                       │
       └─────┬───────┘                       │
             ▼                               │
        [generate] (Step 4) ◄────────────────┘
             │
             ▼
            END

설계 의도:
- 라우터로 진입점을 분기 → 도메인별 다른 처리 가능 (Step 8)
- RAG 경로는 항상 grade로 한 번 거르고, 부족하면 웹 검색으로 보강 (Step 5: CRAG 패턴)
- generate가 모든 경로의 종착지 → 답변 생성 로직은 한 곳에서 관리
"""
from langgraph.graph import StateGraph, END

from services.chatbot.state import GraphState
from services.chatbot.nodes import (
    retrieve,
    grade_documents,
    web_search,
    generate,
    route_question,
    route_question_edge,
)


def decide_after_grade(state: GraphState) -> str:
    """grade_documents 이후 분기를 결정하는 조건부 엣지

    - 관련 문서 부족(web_search_needed=='yes') → 웹 검색으로 보강
    - 충분 → 바로 답변 생성
    """
    if state.get("web_search_needed") == "yes":
        return "web_search"
    return "generate"


def create_graph():
    """그래프 생성 — 노드/엣지를 모두 연결한 컴파일 가능한 워크플로우 반환"""
    workflow = StateGraph(GraphState)

    # 1) 노드 등록
    workflow.add_node("route_question", route_question)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)

    # 2) 진입점: 라우터
    workflow.set_entry_point("route_question")

    # 3) 라우터 → 조건부 분기
    workflow.add_conditional_edges(
        "route_question",
        route_question_edge,
        {
            "retrieve": "retrieve",     # hairstyle, app_usage
            "web_search": "web_search", # general
        },
    )

    # 4) RAG 경로
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grade,
        {
            "web_search": "web_search",
            "generate": "generate",
        },
    )

    # 5) 웹 검색 후 답변 생성
    workflow.add_edge("web_search", "generate")

    # 6) 종료
    workflow.add_edge("generate", END)

    return workflow.compile()


# 직접 실행: 샘플 질문으로 end-to-end 테스트
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    app = create_graph()

    test_questions = [
        "둥근 얼굴에 어울리는 머리는?",      # hairstyle → RAG
        "무료로 몇 번 쓸 수 있어?",          # app_usage → RAG
        "오늘 서울 날씨 어때?",              # general → 웹 검색
    ]

    for q in test_questions:
        print("\n" + "=" * 60)
        print(f"질문: {q}")
        print("=" * 60)
        result = app.invoke({"question": q})
        print(f"\n[최종 답변]\n{result['generation']}")
