"""탈퇴 후 재가입 체험 재지급 방지 (Pro 계정)

Pro 계정이 탈퇴하면 카카오 회원번호를 되돌릴 수 없는 값(HMAC-SHA256)으로 바꿔
PRO_WITHDRAWAL_HOLD_DAYS(약 6개월) 동안만 보관한다. 그 기간에 같은 카카오 계정으로
다시 가입하면 가입은 되지만 가입 체험(PRO_SIGNUP_BONUS_CREDITS)을 주지 않는다.

저장 위치: hairstyle_usage 테이블 (TTL 속성 expire_at 을 이미 쓰는 테이블, README 참고)
- device_id = "withdrawn_kakao#<HMAC hex>", date = "withdrawn"
- expire_at = 탈퇴 시각 + 보관 기간 (epoch 초) → DynamoDB TTL 이 자동 삭제
- TTL 삭제는 늦어질 수 있으므로(최대 약 48시간) 조회할 때도 expire_at 을 직접 비교한다.

HMAC 키는 서버 비밀값(JWT_SECRET_KEY)에서 용도를 붙여 따로 만든다. 비밀값 없이는
보관 값에서 회원번호를 되찾거나 대입해 맞춰 볼 수 없다. 비밀값을 바꾸면 그 전 보관 값은
더 이상 맞지 않는다 (체험이 다시 지급될 수 있다 - 의도된 한계).
"""

import hashlib
import hmac
import os
import time
from typing import Any, Dict

from config.settings import settings
from services.usage_limit_service import get_usage_limit_service

HOLD_KEY_PREFIX = "withdrawn_kakao#"
HOLD_SORT_KEY = "withdrawn"
_HMAC_PURPOSE = b"hairme-pro-withdrawal-hold:v1:"


def _hmac_key() -> bytes:
    secret = settings.JWT_SECRET_KEY
    if not secret:
        raise RuntimeError("JWT_SECRET_KEY 가 없어 탈퇴 보관 값을 만들 수 없습니다")
    return hashlib.sha256(_HMAC_PURPOSE + secret.encode("utf-8")).digest()


def hold_key(kakao_id: str) -> str:
    """카카오 회원번호 → 되돌릴 수 없는 보관 키"""
    digest = hmac.new(_hmac_key(), str(kakao_id).encode("utf-8"), hashlib.sha256)
    return f"{HOLD_KEY_PREFIX}{digest.hexdigest()}"


def _usage_table_name() -> str:
    return os.getenv("DYNAMODB_USAGE_TABLE_NAME", settings.DYNAMODB_USAGE_TABLE_NAME)


def hold_transact_item(kakao_id: str, now: float = None) -> Dict[str, Any]:
    """회원 삭제 트랜잭션에 함께 넣을 보관 기록 (UpdateItem - 테이블이 이미 쓰는 권한)"""
    now = time.time() if now is None else now
    expire_at = int(now + settings.PRO_WITHDRAWAL_HOLD_DAYS * 86400)
    return {
        "Update": {
            "TableName": _usage_table_name(),
            "Key": {"device_id": hold_key(kakao_id), "date": HOLD_SORT_KEY},
            "UpdateExpression": "SET expire_at = :exp, withdrawn_at = :now",
            "ExpressionAttributeValues": {":exp": expire_at, ":now": int(now)},
        }
    }


def recently_withdrawn(kakao_id: str) -> bool:
    """보관 기간 안에 탈퇴한 카카오 계정인지 (조회 실패는 예외로 전파)"""
    response = get_usage_limit_service().table.get_item(
        Key={"device_id": hold_key(kakao_id), "date": HOLD_SORT_KEY},
        ConsistentRead=True,
    )
    item = response.get("Item")
    if not item:
        return False
    return int(item.get("expire_at", 0)) > time.time()
