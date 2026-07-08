"""
벡터DB 구축 및 검색 모듈

문서를 청크로 나누고 → 임베딩(숫자 벡터)으로 변환 → FAISS에 저장
검색 시: 질문도 임베딩으로 변환 → 가장 가까운 벡터를 가진 문서 반환

면접 포인트:
"텍스트를 벡터로 변환해서 의미 기반 검색(semantic search)을 합니다.
키워드 매칭이 아니라, '둥근 얼굴'과 '라운드 페이스'처럼 의미가 같으면
찾아주는 구조입니다."
"""
import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 문서 경로
DATA_DIR = Path(__file__).parent / "data"
VECTORSTORE_DIR = Path(__file__).parent / "data" / "vectorstore"


def build_vectorstore() -> FAISS:
    """문서를 로드하고 벡터DB를 구축

    과정:
    1. txt 파일 로드
    2. 청크로 분할 (왜? LLM 컨텍스트 제한 + 검색 정확도)
    3. 임베딩 변환 (텍스트 → 숫자 벡터)
    4. FAISS에 저장
    """
    # 1. 문서 로드
    docs = []
    for txt_file in DATA_DIR.glob("*.txt"):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        docs.extend(loader.load())

    print(f"로드된 문서 수: {len(docs)}")

    # 2. 청크 분할
    # chunk_size=200: 한 청크가 약 200자 (한국어 기준 2~3문장)
    # chunk_overlap=50: 청크 사이 50자 겹침 (문맥 유지)
    # → 왜 나누나? 전체 문서를 통으로 넣으면 관련 없는 내용까지 포함됨
    # → 작게 나눠야 "이 질문에 딱 맞는 부분"만 검색 가능
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )
    splits = text_splitter.split_documents(docs)
    print(f"분할된 청크 수: {len(splits)}")

    # 3~4. 임베딩 + FAISS 저장
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(splits, embeddings)

    # 로컬에 저장 (다음에 다시 빌드 안 해도 됨)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f"벡터DB 저장 완료: {VECTORSTORE_DIR}")

    return vectorstore


def load_vectorstore() -> FAISS:
    """저장된 벡터DB 로드 (이미 빌드된 경우)"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_retriever(k: int = 3):
    """검색기(retriever) 반환

    k=3: 상위 3개 문서를 반환
    → 왜 3개? 너무 적으면 관련 정보를 놓치고, 너무 많으면 노이즈가 섞임
    """
    if VECTORSTORE_DIR.exists():
        vectorstore = load_vectorstore()
    else:
        vectorstore = build_vectorstore()

    return vectorstore.as_retriever(search_kwargs={"k": k})


# 직접 실행 시: 벡터DB 빌드 + 검색 테스트
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=== 벡터DB 빌드 ===")
    vectorstore = build_vectorstore()

    print("\n=== 검색 테스트 ===")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 테스트 질문들
    test_questions = [
        "둥근 얼굴에 어울리는 머리는?",
        "퍼스널컬러 분석은 어떻게 하나요?",
        "무료로 몇 번 쓸 수 있어?",
    ]

    for q in test_questions:
        print(f"\n질문: {q}")
        results = retriever.invoke(q)
        for i, doc in enumerate(results):
            print(f"  [{i+1}] {doc.page_content[:80]}...")
