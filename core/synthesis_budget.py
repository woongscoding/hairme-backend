"""전체 호출량을 기준으로 비로그인 신규 합성을 제한하는 장치

이것은 "전체 비용의 하드 상한"이 아니다. 아래 세 가지 이유로 상한을 넘겨서
비용이 나갈 수 있고, 그것이 설계상 의도된 동작이다:

  1) 호출 후 집계다. 판정은 이미 기록된 값으로 하고 증가는 호출이 끝난 뒤에
     하므로, 동시에 들어온 요청들은 모두 통과한 뒤에야 합산된다
     (상한 근처에서 동시 요청 수만큼 초과할 수 있다).
  2) 회원은 아예 제한하지 않는다. 상한에 닿아도 회원 합성은 계속 나간다.
  3) 집계 저장소 장애 시 통과시킨다(fail-open). 못 읽으면 막지 않는다.

즉 "비로그인 신규 합성을 멈추는 브레이크"이지, 지출을 특정 금액에서 끊는
장치가 아니다. 실제 지출 보호가 필요하면 AWS Budgets 등 외부 수단을 함께 쓸 것.

무엇을 세는가:
- 캐시 히트가 아닌 실제 Gemini 호출 횟수(api_calls, 재시도 포함)를 하루 단위로
  합산한다. 요청 수가 아니라 호출 수라서, 재시도로 3배가 나간 요청도 3으로 센다.
- 회원/비로그인을 가리지 않고 모두 합산한다 (제한은 비로그인만, 집계는 전체).
- 사전 거절(중복 요청 409 / 예산 503 / 한도 402·429)은 Gemini 를 부르지 않으므로
  집계에 들어가지 않는다. 그런 이벤트의 구조화 로그에는 api_calls 필드 자체를
  넣지 않아, 로그를 합산할 때도 섞이지 않는다.
- 이 모듈은 api/endpoints/synthesis.py 의 두 엔드포인트만 집계한다.
  /api/hair-color/synthesize 도 Gemini 이미지 생성을 호출하지만 아직 포함되지
  않는다 - "하루 전체"는 헤어스타일 합성 두 경로 기준이다.

누구를 막는가:
- 비로그인 경로만 막는다. 회원은 상한과 무관하게 전원 통과시킨다
  (크레딧 잔액은 hairme-users.credits 단일 정수라 유료/보너스 출처를 나눌 수
  없다. 출처 구분이 가능해지기 전까지는 회원을 막지 않는 쪽을 택했다).
- settings.DAILY_SYNTHESIS_BUDGET 이 0(기본)이면 상한 조회를 건너뛴다.
  집계(record_api_calls)는 상한값과 무관하게 항상 수행한다 - 상한을 정하려면
  먼저 평소 호출량을 모아야 하기 때문이다.

어디에 저장하는가:
- 새 테이블을 만들지 않고 기존 hairstyle_usage 를 재사용한다.
  파티션 키 "budget#gemini_calls", 정렬 키는 KST 날짜라 하루에 한 행이다.
  ("ip#", "reward_ad#", "synlock#" 과 같은 서버 전용 네임스페이스 방식이고,
  클라이언트 device_id 는 '#' 이 금지되어 키가 겹치지 않는다)
- 증가는 조건 없는 원자적 UpdateItem 이다. 상한을 넘겨도 실제 호출 수를 그대로
  기록해야 얼마나 초과했는지 볼 수 있다.

날짜 경계와 TTL 은 다른 이야기다:
- 날짜가 바뀌면 정렬 키가 바뀌므로 그날의 집계는 새 행에서 0 부터 시작한다.
  어제 행은 그대로 남아 있고, 오늘 판정에 쓰이지 않을 뿐이다.
- expire_at(TTL)은 그 남은 행을 언젠가 지우기 위한 것이다. DynamoDB TTL 은
  지정 시각에 즉시 삭제하지 않는다. 보통 수 시간, 최대 48시간까지 늦어질 수
  있고 삭제 시점은 보장되지 않는다. 따라서 TTL 에 의존해 "자정에 초기화된다"고
  말하면 안 된다. 초기화는 날짜 키가 바꾸고, TTL 은 청소만 한다.
- 지난 날짜의 집계를 며칠 뒤에 비교하려면 이 행이 남아 있으리라 기대하지 말고
  CloudWatch 로그를 쓸 것 (docs/OPS_SYNTHESIS_COST_METRICS.md).

장애 정책:
- 집계 저장소가 죽으면 합성을 막지 않는다(fail-open). core.synthesis_lock 과 같다.
"""

from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from config.settings import settings
from core.logging import log_structured, logger
from services.usage_limit_service import KST, get_usage_limit_service

# 일 예산 카운터의 파티션 키 (정렬 키는 KST 날짜)
BUDGET_KEY = "budget#gemini_calls"


def _today_kst() -> str:
    """오늘 날짜 (KST, YYYY-MM-DD) - hairstyle_usage 의 정렬 키 규칙과 동일"""
    return datetime.now(KST).strftime("%Y-%m-%d")


def _tomorrow_kst_epoch() -> int:
    """내일 자정 KST (TTL용) - hairstyle_usage 의 expire_at 규칙과 동일"""
    tomorrow = (datetime.now(KST) + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return int(tomorrow.timestamp())


def _table(factory: Optional[Callable[[], Any]]):
    return (factory or get_usage_limit_service)().table


def daily_budget_exceeded(
    *,
    endpoint: str = "unknown",
    usage_service_factory: Optional[Callable[[], Any]] = None,
) -> bool:
    """
    오늘 지금까지 기록된 Gemini 호출 수가 상한에 닿았는지.

    "지금까지 기록된" 이다. 진행 중인 호출은 아직 더해지지 않았으므로 이 판정은
    항상 과소 추정이고, 동시 요청은 함께 통과할 수 있다 (모듈 docstring 참고).

    호출부는 비로그인 경로에서만 이 결과로 거절해야 한다 (회원은 전원 통과).
    잠금 획득보다 먼저 확인한다 - 거절될 요청이 잠금을 잡으면 안 된다.

    Returns:
        True: 상한 도달 (비로그인 합성을 503 으로 거절)
        False: 여유 있음, 상한 미설정(0), 또는 집계 저장소 장애(통과시킴)
    """
    limit = settings.DAILY_SYNTHESIS_BUDGET
    if limit <= 0:
        return False

    try:
        table = _table(usage_service_factory)
        response = table.get_item(Key={"device_id": BUDGET_KEY, "date": _today_kst()})
        used = int((response.get("Item") or {}).get("count", 0))
    except Exception as e:
        # 집계를 못 읽었다고 합성을 막지는 않는다
        logger.warning(f"⚠️ 일 합성 예산 조회 실패 (통과시킴): {str(e)}")
        log_structured(
            "synthesis_budget_unavailable",
            {"endpoint": endpoint, "operation": "read", "error_message": str(e)},
        )
        return False

    if used >= limit:
        logger.warning(f"일 합성 예산 상한 도달: used={used}/{limit}")
        return True
    return False


def record_api_calls(
    api_calls: int,
    *,
    endpoint: str = "unknown",
    usage_service_factory: Optional[Callable[[], Any]] = None,
) -> None:
    """
    실제 Gemini 호출 수를 오늘 집계에 더한다 (원자적 증가, best effort).

    settings.DAILY_SYNTHESIS_BUDGET 값과 무관하게 항상 기록한다.
    상한이 0(무제한)이어도 집계는 계속돼야 상한을 정할 근거가 쌓인다.

    캐시 히트와 사전 거절은 호출이 없으므로 호출부가 아예 부르지 않는다.
    합성 성공/실패 모두 비용은 이미 나갔으므로 둘 다 기록한다 (재시도 포함).
    """
    if api_calls <= 0:
        return

    try:
        table = _table(usage_service_factory)
        table.update_item(
            Key={"device_id": BUDGET_KEY, "date": _today_kst()},
            # 상한을 넘겨도 조건 없이 기록한다 (얼마나 초과했는지 보여야 한다)
            UpdateExpression="SET #cnt = if_not_exists(#cnt, :zero) + :inc, #ttl = :ttl",
            ExpressionAttributeNames={"#cnt": "count", "#ttl": "expire_at"},
            ExpressionAttributeValues={
                ":zero": 0,
                ":inc": api_calls,
                ":ttl": _tomorrow_kst_epoch(),
            },
        )
    except Exception as e:
        # 집계 실패가 합성 응답을 깨뜨리면 안 된다
        logger.warning(f"⚠️ 일 합성 예산 집계 실패 (무시): {str(e)}")
        log_structured(
            "synthesis_budget_unavailable",
            {"endpoint": endpoint, "operation": "record", "error_message": str(e)},
        )


def budget_exceeded_response_body() -> dict:
    """503 응답 본문 (앱이 문구를 그대로 표시한다)"""
    return {
        "success": False,
        "error": "daily_budget_exceeded",
        "message": (
            "오늘 무료 합성이 많이 몰려 잠시 제한되고 있어요. "
            "로그인하시면 바로 이용할 수 있고, 내일 다시 시도하셔도 됩니다."
        ),
    }
