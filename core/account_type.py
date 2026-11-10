"""계정 유형 (손님용 HairMe / 미용실용 Pro)

계정 유형은 로그인할 때 "어느 카카오 앱이 발급한 토큰인지"로 서버가 정한다
(services/kakao_auth_service.py). 클라이언트가 보내는 값으로는 정하지 않는다.

- 회원 행: account_type 속성. 손님 계정은 속성을 쓰지 않는다 (기존 회원과 같은 모양).
- JWT: Pro 계정만 "acct": "pro" 클레임을 넣는다. 클레임이 없으면 손님으로 본다.
"""

from typing import Any, Mapping, Optional

ACCOUNT_CUSTOMER = "customer"
ACCOUNT_PRO = "pro"

# JWT 클레임 이름
ACCOUNT_TYPE_CLAIM = "acct"


def normalize_account_type(value: Any) -> str:
    """알 수 없는 값·없는 값은 전부 손님으로 본다"""
    return ACCOUNT_PRO if value == ACCOUNT_PRO else ACCOUNT_CUSTOMER


def is_pro(account_type: Optional[str]) -> bool:
    return account_type == ACCOUNT_PRO


def account_type_of_user(user: Optional[Mapping[str, Any]]) -> str:
    """회원 행의 계정 유형 (속성이 없는 기존 회원은 손님)"""
    return normalize_account_type((user or {}).get("account_type"))


def account_type_of_payload(payload: Optional[Mapping[str, Any]]) -> str:
    """JWT 페이로드의 계정 유형 (클레임이 없는 기존 토큰은 손님)"""
    return normalize_account_type((payload or {}).get(ACCOUNT_TYPE_CLAIM))
