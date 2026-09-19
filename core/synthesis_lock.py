"""합성 중복 요청 방지 잠금 (DynamoDB 조건부 쓰기)

같은 주체(user_id 또는 device_id)가 같은 cache_key 로 동시에 합성을 요청하면
두 번째 요청을 거절해 Gemini 호출 비용이 배로 나가는 것을 막는다.

왜 DynamoDB 인가:
- Lambda 는 인스턴스가 여러 개라 in-process 잠금이 통하지 않는다.
- 새 테이블을 만들지 않고 기존 hairstyle_usage 테이블을 재사용한다.
  이 테이블은 이미 "ip#<addr>", "reward_ad#<user_id>" 같은 서버 전용
  네임스페이스 키를 파티션 키에 함께 쓰고 있고(core.quota 참고),
  클라이언트가 보낸 device_id 는 '#' 이 금지되어 키가 겹치지 않는다.
  주체 식별자는 해시로만 넣으므로 원문이 테이블 키에 남지도 않는다.

왜 TTL 속성만으로는 안 되는가:
- DynamoDB TTL 삭제는 최대 48시간까지 늦어질 수 있어 120초 잠금의 만료 기준으로
  쓸 수 없다. 만료는 아이템의 lock_expires_at 을 조건식에서 직접 비교해 판정하고,
  expire_at(TTL)은 버려진 잠금을 언젠가 청소하는 용도로만 둔다.

왜 UpdateItem 만 쓰는가:
- 획득도 해제도 조건부 UpdateItem 으로 처리한다. 이 테이블에 대한 기존 코드가
  이미 UpdateItem/GetItem 만 쓰므로, PutItem/DeleteItem 을 새로 쓰면 Lambda 실행
  역할에 권한이 없을 때 잠금이 반쪽만 동작할 수 있다(획득은 되는데 해제가 안 되면
  같은 사용자가 120초 동안 막힌다). IAM 변경 없이 동작하도록 맞췄다.
- 해제는 아이템을 지우지 않고 lock_expires_at 을 0 으로 내려 "만료" 상태로 만든다.
  해제 즉시 다음 요청이 통과할 수 있고, 남은 행은 expire_at(TTL)이 나중에
  청소한다. 잠금이 풀리는 시점과 행이 사라지는 시점은 별개다.

장애 정책:
- 잠금 저장소가 죽으면 합성을 막지 않는다(fail-open). 중복 과금보다 서비스
  중단이 더 나쁘기 때문. 대신 경고 로그를 남긴다.
"""

import hashlib
import time
from typing import Any, Callable, Optional, Tuple

try:
    from botocore.exceptions import ClientError
except ImportError:  # boto3 미설치 환경 (로컬 테스트)
    ClientError = None  # type: ignore[assignment]

from core.logging import log_structured, logger
from services.usage_limit_service import get_usage_limit_service

# 잠금 유지 시간. 합성 1회(재시도 3회 포함)가 끝나기에 충분하면서,
# 프로세스가 죽어 해제되지 않은 잠금이 오래 남지 않을 만큼 짧아야 한다.
LOCK_TTL_SECONDS = 120

# 버려진 잠금 아이템의 TTL (조건식이 아니라 청소용)
LOCK_ITEM_TTL_SECONDS = 3600

# 잠금 아이템의 정렬 키. 날짜를 쓰면 자정 경계에서 키가 갈려 같은 요청이
# 두 번 통과하므로 고정값을 쓴다.
LOCK_SORT_KEY = "lock"

NOOP_RELEASE: Callable[[], None] = lambda: None

LockResult = Tuple[bool, Callable[[], None]]


def build_lock_key(subject: str, cache_key: str) -> str:
    """잠금 파티션 키: "synlock#<주체 해시>#<cache_key 앞 32자>"

    주체(user_id/device_id) 원문은 테이블에 남기지 않는다.
    """
    subject_hash = hashlib.sha256(subject.encode("utf-8")).hexdigest()[:16]
    return f"synlock#{subject_hash}#{cache_key[:32]}"


def acquire_synthesis_lock(
    subject: Optional[str],
    cache_key: str,
    *,
    endpoint: str = "unknown",
    usage_service_factory: Optional[Callable[[], Any]] = None,
) -> LockResult:
    """
    합성 진행 중 잠금 획득.

    크레딧 차감보다 먼저 호출해야 한다 (거절된 요청이 과금되지 않도록).

    Args:
        subject: user_id 또는 device_id. None 이면 잠글 대상이 없어 통과시킨다
        cache_key: 같은 사진 + 같은 스타일을 구분하는 키
        endpoint: 로그용 엔드포인트 이름
        usage_service_factory: 테스트 주입용 (기본은 hairstyle_usage 싱글톤)

    Returns:
        (acquired, release)
        - acquired=False: 같은 요청이 이미 진행 중 -> 호출부가 409 로 거절한다
        - acquired=True: 진행 가능. 완료/실패 시 release() 를 반드시 호출한다
          (저장소 장애로 잠그지 못한 경우에도 True + no-op release 를 돌려준다)
    """
    if not subject:
        return True, NOOP_RELEASE

    factory = usage_service_factory or get_usage_limit_service
    lock_key = build_lock_key(subject, cache_key)
    now = int(time.time())

    try:
        table = factory().table
        table.update_item(
            Key={"device_id": lock_key, "date": LOCK_SORT_KEY},
            UpdateExpression="SET #lock = :expires, #ttl = :item_ttl",
            ConditionExpression="attribute_not_exists(#lock) OR #lock < :now",
            ExpressionAttributeNames={
                "#lock": "lock_expires_at",
                "#ttl": "expire_at",
            },
            ExpressionAttributeValues={
                ":expires": now + LOCK_TTL_SECONDS,
                ":item_ttl": now + LOCK_ITEM_TTL_SECONDS,
                ":now": now,
            },
        )
    except Exception as e:
        if _is_conditional_check_failure(e):
            logger.warning(f"합성 중복 요청 차단: endpoint={endpoint}")
            return False, NOOP_RELEASE

        # 저장소 장애 - 합성을 막지 않는다 (중복 과금보다 중단이 더 나쁘다)
        logger.warning(f"⚠️ 합성 잠금 확인 실패 (통과시킴): {str(e)}")
        log_structured(
            "synthesis_lock_unavailable",
            {"endpoint": endpoint, "error_message": str(e)},
        )
        return True, NOOP_RELEASE

    def release() -> None:
        """합성 완료/실패 시 잠금 해제 (best effort)

        아이템을 지우지 않고 만료 시각을 0 으로 내린다 (DeleteItem 권한 불필요).
        """
        try:
            table.update_item(
                Key={"device_id": lock_key, "date": LOCK_SORT_KEY},
                UpdateExpression="SET #lock = :zero",
                ExpressionAttributeNames={"#lock": "lock_expires_at"},
                ExpressionAttributeValues={":zero": 0},
            )
        except Exception as release_error:
            # 해제에 실패해도 LOCK_TTL_SECONDS 뒤에는 조건식이 만료로 판정한다
            logger.warning(f"⚠️ 합성 잠금 해제 실패 (무시): {str(release_error)}")

    return True, release


def _is_conditional_check_failure(error: Exception) -> bool:
    """조건부 쓰기 실패(= 이미 진행 중)인지 판별"""
    if ClientError is not None and isinstance(error, ClientError):
        return (
            error.response.get("Error", {}).get("Code")
            == "ConditionalCheckFailedException"
        )
    # boto3 가 없는 환경의 목 객체 대응
    return error.__class__.__name__ == "ConditionalCheckFailedException"


def duplicate_request_response_body() -> dict:
    """409 응답 본문 (앱이 문구를 그대로 표시한다)"""
    return {
        "success": False,
        "error": "synthesis_in_progress",
        "message": "같은 사진으로 합성이 진행 중입니다. 잠시 후 결과를 확인해주세요.",
    }
