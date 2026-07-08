"""
검색 노드 (Retrieve)

역할: State의 question을 받아 벡터DB에서 관련 문서를 검색해서 documents에 채워 넣음

면접 포인트:
"노드는 State를 입력받아 일부 필드를 갱신해서 반환하는 순수 함수에 가깝습니다.
이 단위가 작아야 단위 테스트가 쉽고, 그래프 구성도 레고처럼 조립할 수 있습니다."
"""
from services.chatbot.state import GraphState
from services.chatbot.vectorstore import get_retriever

# 모듈 로드 시 1회만 retriever 생성 (요청마다 벡터DB 로드 비용 회피)
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever(k=3)
    return _retriever


def retrieve(state: GraphState) -> dict:
    """벡터DB 검색 노드

    입력: state["question"]
    출력: {"documents": [문서 텍스트 리스트]}

    LangGraph 규칙: dict로 반환한 키만 State에 머지됨 (부분 업데이트)
    """
    question = state["question"]
    print(f"---RETRIEVE--- 질문: {question}")

    retriever = _get_retriever()
    docs = retriever.invoke(question)

    # Document 객체 → 문자열 리스트 (State는 단순한 게 다루기 쉬움)
    documents = [doc.page_content for doc in docs]
    print(f"---RETRIEVE--- 검색된 문서 수: {len(documents)}")

    return {"documents": documents}
