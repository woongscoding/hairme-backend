"""
답변 생성 노드 (Generate)

역할: 검색된 문서(documents)를 컨텍스트로 LLM이 최종 답변을 생성

면접 포인트:
"프롬프트에서 '주어진 컨텍스트만 사용하라'고 명시해서 환각을 줄입니다.
컨텍스트에 답이 없으면 모른다고 답하도록 가이드해서, '아무 말이나 만들어내는'
일반 챗봇과 차별화합니다."
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from services.chatbot.state import GraphState

_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 HairMe 앱의 친절한 헤어스타일 상담 어시스턴트입니다.\n"
                    "아래에 주어진 컨텍스트를 근거로 사용자 질문에 답변하세요.\n\n"
                    "규칙:\n"
                    "1. 컨텍스트에 있는 정보만 사용하세요. 추측하지 마세요.\n"
                    "2. 컨텍스트에 답이 없으면 솔직하게 '해당 정보가 없습니다'라고 답하세요.\n"
                    "3. 한국어로 친근하지만 정확하게 답변하세요.\n"
                    "4. 답변은 3~5문장으로 간결하게.",
                ),
                (
                    "human",
                    "컨텍스트:\n{context}\n\n질문: {question}",
                ),
            ]
        )
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        _chain = prompt | llm | StrOutputParser()
    return _chain


def generate(state: GraphState) -> dict:
    """LLM 답변 생성

    입력: state["question"], state["documents"]
    출력: {"generation": 생성된 답변}
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state.get("documents", []) or []

    context = "\n\n".join(documents) if documents else "(관련 문서 없음)"
    chain = _get_chain()
    answer = chain.invoke({"context": context, "question": question})

    print(f"  생성된 답변: {answer[:80]}...")
    return {"generation": answer}
