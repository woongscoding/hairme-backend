"""
문서 관련성 평가 노드 (Grade Documents)

역할: 검색된 문서가 질문에 정말 관련 있는지 LLM에게 yes/no로 판단시킴
- 관련 있는 문서만 남기고
- 하나도 관련 없으면 web_search_needed = "yes"로 표시

면접 포인트:
"검색 결과를 그대로 LLM에 넘기면 노이즈가 답변 품질을 떨어뜨립니다.
LLM 자기 자신을 'judge'로 한 번 거쳐서 관련성을 필터링하는 게
Self-RAG / CRAG 계열 패턴입니다. 비용은 들지만 환각(hallucination)을
크게 줄일 수 있습니다."
"""
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from services.chatbot.state import GraphState


class GradeDocument(BaseModel):
    """LLM 출력 스키마 — yes/no 둘 중 하나만 나오도록 강제"""

    binary_score: Literal["yes", "no"] = Field(
        description="문서가 질문과 관련 있으면 'yes', 아니면 'no'"
    )


_grader = None


def _get_grader():
    """structured output을 쓰는 grader 체인 — 1회만 생성"""
    global _grader
    if _grader is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(GradeDocument)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 검색된 문서가 사용자 질문과 관련 있는지 판단하는 평가자입니다.\n"
                    "엄격할 필요는 없습니다 — 문서에 질문과 의미상 연결되는 키워드나 "
                    "주제가 있으면 관련 있다고 판단하세요.\n"
                    "관련 있으면 'yes', 없으면 'no'만 출력하세요.",
                ),
                ("human", "검색된 문서:\n\n{document}\n\n사용자 질문: {question}"),
            ]
        )
        _grader = prompt | structured_llm
    return _grader


def grade_documents(state: GraphState) -> dict:
    """검색된 문서를 하나씩 평가해서 관련 있는 것만 남김

    입력: state["question"], state["documents"]
    출력: {"documents": 필터링된 리스트, "web_search_needed": "yes"|"no"}
    """
    print("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]

    grader = _get_grader()
    filtered = []
    web_search_needed = "no"

    for doc in documents:
        score = grader.invoke({"question": question, "document": doc})
        if score.binary_score == "yes":
            print(f"  [관련 O] {doc[:60]}...")
            filtered.append(doc)
        else:
            print(f"  [관련 X] {doc[:60]}...")

    # 관련 문서가 하나도 없으면 → 웹 검색으로 fallback
    if not filtered:
        print("---관련 문서 0건 → 웹 검색 필요---")
        web_search_needed = "yes"

    return {"documents": filtered, "web_search_needed": web_search_needed}
