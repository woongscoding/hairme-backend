"""
HairMe 챗봇의 State 정의

LangGraph에서 State = 노드 간 공유되는 "화이트보드"
- 모든 노드가 여기서 데이터를 읽고, 처리 결과를 다시 써
- 일반 체인(릴레이 달리기)과 달리, 어떤 노드든 이전 데이터에 접근 가능
- 이게 있어야 "검색 결과가 별로면 다시 검색" 같은 루프가 가능해짐
"""
from typing import List, TypedDict


class GraphState(TypedDict):
    """그래프 전체에서 공유되는 상태

    설계 원칙:
    1. 최소한의 필드만 — 디버깅할 때 추적이 쉬워야 해
    2. TypedDict — 팀원이 "이 필드에 뭘 넣지?" 고민 안 하게
    3. 문자열 플래그 — LLM 판단 결과를 바로 넣을 수 있게

    면접 포인트:
    "State를 TypedDict로 정의해서, 어떤 데이터가 파이프라인을 흐르는지
    코드만 보고 바로 파악할 수 있습니다. HR 챗봇처럼 여러 팀원이
    각자 노드를 개발할 때, 이 State가 계약서(contract) 역할을 합니다."
    """

    # ---- 입력 ----
    question: str               # 사용자 원본 질문

    # ---- 라우팅 ----
    question_type: str          # 질문 유형: "hairstyle" | "app_usage" | "general"
                                # → 라우팅 노드가 판단해서 여기에 씀
                                # → 조건부 엣지가 이 값을 보고 분기

    # ---- 검색 ----
    documents: List[str]        # 벡터DB에서 검색된 문서들
                                # → 검색 노드가 씀, 관련성 체크 노드가 읽음

    # ---- 판단 플래그 ----
    web_search_needed: str      # 웹 검색 필요 여부: "yes" | "no"
                                # → 관련성 체크 노드가 씀
                                # → 조건부 엣지가 "yes"면 웹 검색으로, "no"면 생성으로

    # ---- 출력 ----
    generation: str             # LLM이 생성한 최종 답변
