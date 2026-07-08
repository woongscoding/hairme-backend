"""
질문 라우팅 노드 (Route Question)

역할: 사용자 질문을 LLM이 분류해서 question_type에 기록
- "hairstyle" : 헤어/얼굴형/퍼스널컬러 관련 → 벡터DB(RAG)
- "app_usage" : HairMe 앱 사용법 → 벡터DB(RAG)
- "general"   : 그 외 일반 질문 → 웹 검색

면접 포인트:
"라우팅을 그래프 진입부에 두면, 질문 종류별로 완전히 다른 서브 파이프라인을
구성할 수 있습니다. SK하이닉스 HR 챗봇이라면 '근태/복지/사내제도/일반잡담'으로
라우팅해서 각자 다른 인덱스를 타게 할 수 있고, 도메인 추가도 그래프 변경 없이
라우터 카테고리만 늘리면 되므로 확장성이 좋습니다."
"""
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from services.chatbot.state import GraphState


class RouteQuery(BaseModel):
    """라우터 출력 — 셋 중 하나만"""

    datasource: Literal["hairstyle", "app_usage", "general"] = Field(
        description="질문 카테고리"
    )


_router = None


def _get_router():
    global _router
    if _router is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(RouteQuery)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 HairMe 앱 사용자의 질문을 분류하는 라우터입니다.\n"
                    "다음 세 카테고리 중 하나를 선택하세요:\n\n"
                    "- hairstyle: 헤어스타일, 얼굴형, 퍼스널컬러, 염색, 모질 등 미용 도메인 질문\n"
                    "- app_usage: HairMe 앱 사용법, 기능, 결제, 분석 방법, 오류 해결 등\n"
                    "- general: 그 외 일반 상식, 시사, 위 두 카테고리에 속하지 않는 질문\n",
                ),
                ("human", "{question}"),
            ]
        )
        _router = prompt | structured_llm
    return _router


def route_question(state: GraphState) -> dict:
    """질문 분류 노드 (State에 question_type만 채움)

    LangGraph 팁: 노드 자체는 값만 쓰고, '어디로 갈지'는 조건부 엣지 함수가 결정.
    이 분리 덕분에 노드는 재사용 가능하고, 라우팅 규칙은 한 곳에서 관리됨.
    """
    print("---ROUTE QUESTION---")
    question = state["question"]
    router = _get_router()
    result = router.invoke({"question": question})
    print(f"  → {result.datasource}")
    return {"question_type": result.datasource}


def route_question_edge(state: GraphState) -> str:
    """조건부 엣지용 함수 — 다음 노드 이름을 문자열로 반환

    이렇게 노드 함수와 엣지 함수를 분리하면 그래프 정의(graph.py)에서
    '어떤 분기 키가 어디로 가는지'를 명시적으로 매핑할 수 있어 가독성이 좋음.
    """
    qtype = state["question_type"]
    if qtype == "general":
        return "web_search"
    # hairstyle, app_usage 모두 같은 RAG 파이프라인 사용
    return "retrieve"
