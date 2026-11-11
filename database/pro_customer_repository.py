"""Pro 계정(미용실) 손님 기록 리포지토리 (DynamoDB hairme-pro-customers 테이블)

테이블 구조 (scripts/create_pro_customers_table.py):
- Partition Key: user_id (String, 디자이너의 user_id)
- Sort Key: sk (String)
- TTL 속성: expire_at (epoch 초)

한 디자이너의 파티션에 세 종류 아이템을 둔다.
- 손님:        sk = "C#{customer_id}"
               name, phone(숫자만), memo, created_at, last_visit_at(없을 수 있음),
               visit_count(N), expire_at
- 전화번호 마커: sk = "P#{phone}", customer_id, expire_at (손님과 같은 값)
               같은 디자이너의 두 손님이 같은 번호를 갖지 못하게 트랜잭션 조건으로 쓴다.
- 상담 기록:   sk = "V#{customer_id}#{visit_id}"
               at, gender, chosen_style(없을 수 있음),
               results: [{result_id, style, photo_key}] (올린 순서), expire_at = at + 보관 기간

보관 기간 (settings.PRO_RECORD_RETENTION_DAYS, 2년):
- 상담 기록은 상담일(at)부터 센다.
- 손님·전화번호 마커는 마지막 상담일부터 센다 (상담이 없으면 등록일부터).
  상담 결과가 저장될 때마다 늘린다.
- DynamoDB TTL 삭제는 최대 약 48시간 늦을 수 있으므로 읽을 때도 expire_at 이
  지난 아이템은 없는 것으로 본다.

이 모듈은 이름·전화번호·메모를 로그에 남기지 않는다.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

try:
    import boto3
    from boto3.dynamodb.conditions import Key
    from botocore.exceptions import ClientError
    from botocore.config import Config

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from config.settings import settings
from core.logging import logger, mask_user_id

CUSTOMER_PREFIX = "C#"
PHONE_PREFIX = "P#"
VISIT_PREFIX = "V#"

# 전화번호 마커 충돌 시 시도 횟수 (주인 없는 마커를 확인한 뒤 한 번 더 시도)
_TRANSACTION_RETRY = 2


def customer_sk(customer_id: str) -> str:
    return f"{CUSTOMER_PREFIX}{customer_id}"


def phone_sk(phone: str) -> str:
    return f"{PHONE_PREFIX}{phone}"


def visit_sk(customer_id: str, visit_id: str) -> str:
    return f"{VISIT_PREFIX}{customer_id}#{visit_id}"


def visits_sk_prefix(customer_id: str) -> str:
    return f"{VISIT_PREFIX}{customer_id}#"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def expire_at_after(moment: datetime) -> int:
    """moment 부터 보관 기간이 끝나는 시각 (epoch 초, DynamoDB TTL 형식)"""
    return int(
        (moment + timedelta(days=settings.PRO_RECORD_RETENTION_DAYS)).timestamp()
    )


def is_expired(item: Optional[Dict[str, Any]], now: datetime) -> bool:
    """TTL 이 아직 지우지 않았어도 보관 기간이 지난 아이템은 없는 것으로 본다"""
    if not item:
        return True
    expire_at = item.get("expire_at")
    if expire_at is None:
        return False
    try:
        return int(expire_at) <= int(now.timestamp())
    except (TypeError, ValueError):
        return False


class RecordsTableMissing(Exception):
    """손님 기록 테이블이 설정되지 않았거나 아직 만들어지지 않았다"""


class DuplicatePhoneError(Exception):
    """같은 디자이너의 다른 손님이 이미 이 전화번호를 쓰고 있다"""

    def __init__(self, customer: Dict[str, Any]):
        self.customer = customer
        super().__init__("이미 등록된 전화번호입니다")


class CustomerGoneError(Exception):
    """손님이 없거나 (동시에) 삭제됐다"""


class VisitChangedError(Exception):
    """상담 기록이 그 사이 바뀌었다 (다시 읽고 시도해야 한다)"""


def _plain(item: Dict[str, Any]) -> Dict[str, Any]:
    """DynamoDB Decimal 을 int 로 바꾼다"""
    plain = dict(item)
    for key in ("visit_count", "expire_at"):
        if key in plain and plain[key] is not None:
            plain[key] = int(plain[key])
    if "results" in plain:
        plain["results"] = [dict(r) for r in plain.get("results") or []]
    return plain


def _cancellation_codes(error: "ClientError") -> List[Optional[str]]:
    reasons = error.response.get("CancellationReasons") or []
    return [reason.get("Code") for reason in reasons]


def _error_code(error: "ClientError") -> str:
    return error.response.get("Error", {}).get("Code", "")


class ProCustomerRepository:
    """손님·전화번호 마커·상담 기록 CRUD (DynamoDB)"""

    def __init__(self):
        self._table = None

    @property
    def table_name(self) -> str:
        return os.getenv(
            "DYNAMODB_PRO_CUSTOMERS_TABLE_NAME",
            settings.DYNAMODB_PRO_CUSTOMERS_TABLE_NAME,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.table_name) and BOTO3_AVAILABLE

    @property
    def table(self):
        """Lazy load the DynamoDB table resource"""
        if self._table is None:
            if not self.enabled:
                raise RecordsTableMissing()
            aws_region = os.getenv("AWS_REGION", settings.AWS_REGION)
            config = Config(
                connect_timeout=5, read_timeout=10, retries={"max_attempts": 3}
            )
            resource = boto3.resource("dynamodb", region_name=aws_region, config=config)
            self._table = resource.Table(self.table_name)
        return self._table

    # ========== 내부 도우미 ==========

    def _call(self, fn, *args, **kwargs):
        """테이블이 아직 없으면 RecordsTableMissing 으로 바꿔 올린다"""
        try:
            return fn(*args, **kwargs)
        except ClientError as e:
            if _error_code(e) == "ResourceNotFoundException":
                raise RecordsTableMissing() from e
            raise

    def _get(self, user_id: str, sk: str) -> Optional[Dict[str, Any]]:
        response = self._call(
            self.table.get_item,
            Key={"user_id": user_id, "sk": sk},
            ConsistentRead=True,
        )
        item = response.get("Item")
        return _plain(item) if item else None

    def _query_prefix(self, user_id: str, prefix: str = "") -> List[Dict[str, Any]]:
        """파티션에서 sk 접두사가 같은 아이템 전부 (prefix 가 비면 파티션 전체)"""
        condition = Key("user_id").eq(user_id)
        if prefix:
            condition = condition & Key("sk").begins_with(prefix)
        kwargs: Dict[str, Any] = {
            "KeyConditionExpression": condition,
            "ConsistentRead": True,
        }
        items: List[Dict[str, Any]] = []
        while True:
            response = self._call(self.table.query, **kwargs)
            items.extend(_plain(item) for item in response.get("Items", []))
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                return items
            kwargs["ExclusiveStartKey"] = last_key

    def _transact(self, items: List[Dict[str, Any]]) -> None:
        # 리소스의 내부 클라이언트는 파이썬 값을 자동 변환한다 (user_repository 주석 참고)
        self._call(self.table.meta.client.transact_write_items, TransactItems=items)

    def _marker_put(
        self,
        user_id: str,
        phone: str,
        customer_id: str,
        expire_at: int,
        now: datetime,
        stale_owner: Optional[str] = None,
    ) -> Dict[str, Any]:
        """전화번호 마커 쓰기. 빈 자리·보관 기간 지난 마커·자기 마커만 덮어쓴다"""
        condition = (
            "attribute_not_exists(sk) OR expire_at <= :now OR customer_id = :self"
        )
        values: Dict[str, Any] = {":now": int(now.timestamp()), ":self": customer_id}
        if stale_owner:
            # 주인 손님이 이미 없는 마커 (지난 삭제가 중간에 끊긴 경우)
            condition += " OR customer_id = :stale"
            values[":stale"] = stale_owner
        return {
            "Put": {
                "TableName": self.table.name,
                "Item": {
                    "user_id": user_id,
                    "sk": phone_sk(phone),
                    "customer_id": customer_id,
                    "expire_at": expire_at,
                },
                "ConditionExpression": condition,
                "ExpressionAttributeValues": values,
            }
        }

    def _phone_owner(
        self, user_id: str, phone: str, now: datetime
    ) -> Optional[Dict[str, Any]]:
        """마커가 가리키는 살아 있는 손님. 마커가 없거나 주인이 없으면 None"""
        marker = self._get(user_id, phone_sk(phone))
        if marker is None or is_expired(marker, now):
            return None
        return self.get_customer(user_id, marker.get("customer_id") or "", now)

    def _stale_marker_owner(self, user_id: str, phone: str) -> Optional[str]:
        marker = self._get(user_id, phone_sk(phone))
        return (marker or {}).get("customer_id")

    # ========== 손님 ==========

    def list_customers(self, user_id: str, now: datetime) -> List[Dict[str, Any]]:
        """보관 기간 안의 손님 전부 (정렬은 호출자)"""
        items = self._query_prefix(user_id, CUSTOMER_PREFIX)
        return [item for item in items if not is_expired(item, now)]

    def get_customer(
        self, user_id: str, customer_id: str, now: datetime
    ) -> Optional[Dict[str, Any]]:
        if not customer_id:
            return None
        item = self._get(user_id, customer_sk(customer_id))
        if item is None or is_expired(item, now):
            return None
        return item

    def create_customer(
        self,
        user_id: str,
        customer_id: str,
        name: str,
        phone: str,
        now: datetime,
    ) -> Dict[str, Any]:
        """손님 + 전화번호 마커를 한 트랜잭션으로 쓴다

        Raises:
            DuplicatePhoneError: 이 디자이너의 다른 손님이 같은 번호를 쓰고 있다
        """
        expire_at = expire_at_after(now)
        item: Dict[str, Any] = {
            "user_id": user_id,
            "sk": customer_sk(customer_id),
            "name": name,
            "phone": phone,
            "memo": "",
            "created_at": to_iso(now),
            "visit_count": 0,
            "expire_at": expire_at,
        }
        stale_owner: Optional[str] = None
        for _ in range(_TRANSACTION_RETRY):
            try:
                self._transact(
                    [
                        {
                            "Put": {
                                "TableName": self.table.name,
                                "Item": item,
                                "ConditionExpression": "attribute_not_exists(sk)",
                            }
                        },
                        self._marker_put(
                            user_id, phone, customer_id, expire_at, now, stale_owner
                        ),
                    ]
                )
                return _plain(item)
            except ClientError as e:
                codes = _cancellation_codes(e)
                if len(codes) < 2 or codes[1] != "ConditionalCheckFailed":
                    raise
            stale_owner = self._resolve_phone_conflict(user_id, phone, now)
        raise RuntimeError("전화번호 마커 정리 후에도 손님을 만들지 못했습니다")

    def _resolve_phone_conflict(self, user_id: str, phone: str, now: datetime) -> str:
        """마커 충돌 처리: 살아 있는 주인이 있으면 DuplicatePhoneError,
        주인이 없는 마커면 그 주인 id 를 돌려준다 (다음 시도에서 덮어쓴다)"""
        owner = self._phone_owner(user_id, phone, now)
        if owner is not None:
            raise DuplicatePhoneError(owner)
        stale = self._stale_marker_owner(user_id, phone) or ""
        logger.warning(
            f"⚠️ 주인 없는 전화번호 마커 정리: user_id={mask_user_id(user_id)}"
        )
        return stale

    def update_customer(
        self,
        user_id: str,
        current: Dict[str, Any],
        changes: Dict[str, str],
        now: datetime,
    ) -> Dict[str, Any]:
        """이름·전화번호·메모 변경. 번호가 바뀌면 마커를 한 트랜잭션에서 바꾼다

        Raises:
            DuplicatePhoneError: 새 번호를 다른 손님이 쓰고 있다
            CustomerGoneError: 그 사이 손님이 삭제됐다
        """
        customer_id = current["sk"][len(CUSTOMER_PREFIX) :]
        changes = {k: v for k, v in changes.items() if current.get(k) != v}
        if not changes:
            return current

        names = {f"#{k}": k for k in changes}
        values: Dict[str, Any] = {f":{k}": v for k, v in changes.items()}
        values[":old_phone"] = current["phone"]
        update = {
            "Update": {
                "TableName": self.table.name,
                "Key": {"user_id": user_id, "sk": customer_sk(customer_id)},
                "UpdateExpression": "SET " + ", ".join(f"#{k} = :{k}" for k in changes),
                # 다른 요청이 그 사이 번호를 바꿨다면 마커가 어긋나지 않게 거절한다
                "ConditionExpression": "attribute_exists(sk) AND phone = :old_phone",
                "ExpressionAttributeNames": names,
                "ExpressionAttributeValues": values,
            }
        }
        new_phone = changes.get("phone")
        stale_owner: Optional[str] = None
        for _ in range(_TRANSACTION_RETRY):
            items = [update]
            if new_phone:
                items.append(
                    self._marker_put(
                        user_id,
                        new_phone,
                        customer_id,
                        int(current["expire_at"]),
                        now,
                        stale_owner,
                    )
                )
                items.append(
                    {
                        "Delete": {
                            "TableName": self.table.name,
                            "Key": {
                                "user_id": user_id,
                                "sk": phone_sk(current["phone"]),
                            },
                            "ConditionExpression": (
                                "attribute_not_exists(sk) OR customer_id = :self"
                            ),
                            "ExpressionAttributeValues": {":self": customer_id},
                        }
                    }
                )
            try:
                self._transact(items)
                return {**current, **changes}
            except ClientError as e:
                codes = _cancellation_codes(e)
                if codes and codes[0] == "ConditionalCheckFailed":
                    raise CustomerGoneError() from e
                if not (
                    new_phone
                    and len(codes) > 1
                    and codes[1] == "ConditionalCheckFailed"
                ):
                    raise
            stale_owner = self._resolve_phone_conflict(user_id, new_phone, now)
        raise RuntimeError("전화번호 마커 정리 후에도 손님을 바꾸지 못했습니다")

    def delete_customer_items(self, user_id: str, customer_id: str) -> bool:
        """손님의 상담 기록 → 전화번호 마커·손님 아이템 순으로 지운다

        손님 아이템이 이미 없어도 남은 상담 기록은 지운다 (지난 삭제가 끊긴 경우).

        Returns:
            손님 아이템이 있었는지
        """
        visits = self._query_prefix(user_id, visits_sk_prefix(customer_id))
        self._batch_delete(user_id, [v["sk"] for v in visits])

        customer = self._get(user_id, customer_sk(customer_id))
        if customer is None:
            return False
        try:
            self._transact(
                [
                    {
                        "Delete": {
                            "TableName": self.table.name,
                            "Key": {"user_id": user_id, "sk": customer["sk"]},
                        }
                    },
                    {
                        "Delete": {
                            "TableName": self.table.name,
                            "Key": {
                                "user_id": user_id,
                                "sk": phone_sk(customer.get("phone") or ""),
                            },
                            # 다른 손님에게 넘어간 번호의 마커는 지우지 않는다
                            "ConditionExpression": (
                                "attribute_not_exists(sk) OR customer_id = :self"
                            ),
                            "ExpressionAttributeValues": {":self": customer_id},
                        }
                    },
                ]
            )
        except ClientError as e:
            codes = _cancellation_codes(e)
            if len(codes) > 1 and codes[1] == "ConditionalCheckFailed":
                self._call(
                    self.table.delete_item,
                    Key={"user_id": user_id, "sk": customer["sk"]},
                )
            else:
                raise
        return True

    def delete_all(self, user_id: str) -> int:
        """디자이너의 파티션 전부 삭제 (회원 탈퇴용)

        Returns:
            지운 손님 수 (보관 기간이 지났지만 아직 남아 있던 손님 포함)
        """
        items = self._query_prefix(user_id)
        self._batch_delete(user_id, [item["sk"] for item in items])
        return sum(1 for item in items if item["sk"].startswith(CUSTOMER_PREFIX))

    def _batch_delete(self, user_id: str, sort_keys: List[str]) -> None:
        if not sort_keys:
            return

        def _run():
            with self.table.batch_writer() as batch:
                for sk in sort_keys:
                    batch.delete_item(Key={"user_id": user_id, "sk": sk})

        self._call(_run)

    # ========== 상담 기록 ==========

    def list_visits(
        self, user_id: str, customer_id: str, now: datetime
    ) -> List[Dict[str, Any]]:
        items = self._query_prefix(user_id, visits_sk_prefix(customer_id))
        return [item for item in items if not is_expired(item, now)]

    def get_visit(
        self, user_id: str, customer_id: str, visit_id: str, now: datetime
    ) -> Optional[Dict[str, Any]]:
        item = self._get(user_id, visit_sk(customer_id, visit_id))
        if item is None or is_expired(item, now):
            return None
        return item

    def add_visit_result(
        self,
        user_id: str,
        customer: Dict[str, Any],
        visit_id: str,
        visit: Optional[Dict[str, Any]],
        result: Dict[str, str],
        gender: str,
        now: datetime,
    ) -> Dict[str, Any]:
        """상담 결과 1건 추가 (상담이 없으면 만든다) + 손님의 마지막 상담일·보관 기간 갱신

        한 트랜잭션: 상담 기록 쓰기 / 손님 갱신 / 전화번호 마커 보관 기간 갱신.

        결과 수 상한은 호출자가 확인한다. 여기서는 읽은 뒤 결과 수가 바뀌지 않았을 때만
        붙이므로(size 조건) 동시 요청이 상한을 넘기지 못한다.

        Args:
            visit: 지금 읽은 상담 기록 (없으면 None → 새로 만든다, at = now)

        Raises:
            VisitChangedError: 그 사이 상담이 만들어졌거나 바뀌었다 (다시 읽고 재시도)
            CustomerGoneError: 그 사이 손님이 삭제됐다
        """
        customer_id = customer["sk"][len(CUSTOMER_PREFIX) :]
        created = visit is None
        visit_at = now if created else (parse_iso(visit.get("at")) or now)
        visit_at_iso = to_iso(now) if created else visit["at"]

        # 손님은 "마지막 상담일 + 보관 기간" 까지 남는다 (더 늦은 상담이 있으면 그대로)
        last_visit = parse_iso(customer.get("last_visit_at"))
        if last_visit is None or visit_at > last_visit:
            last_visit_iso = visit_at_iso
        else:
            last_visit_iso = customer["last_visit_at"]
        expire_at = max(int(customer.get("expire_at") or 0), expire_at_after(visit_at))

        table_name = self.table.name
        key = {"user_id": user_id, "sk": visit_sk(customer_id, visit_id)}
        if created:
            visit_item = {
                **key,
                "at": visit_at_iso,
                "gender": gender,
                "results": [result],
                "expire_at": expire_at_after(visit_at),
            }
            visit_write: Dict[str, Any] = {
                "Put": {
                    "TableName": table_name,
                    "Item": visit_item,
                    "ConditionExpression": "attribute_not_exists(sk) OR expire_at <= :now",
                    "ExpressionAttributeValues": {":now": int(now.timestamp())},
                }
            }
        else:
            visit_write = {
                "Update": {
                    "TableName": table_name,
                    "Key": key,
                    "UpdateExpression": "SET results = list_append(results, :r)",
                    "ConditionExpression": (
                        "attribute_exists(sk) AND size(results) = :count"
                    ),
                    "ExpressionAttributeValues": {
                        ":r": [result],
                        ":count": len(visit.get("results") or []),
                    },
                }
            }

        customer_update = "SET last_visit_at = :last, expire_at = :exp"
        customer_values: Dict[str, Any] = {":last": last_visit_iso, ":exp": expire_at}
        if created:
            customer_update += " ADD visit_count :one"
            customer_values[":one"] = 1

        try:
            self._transact(
                [
                    visit_write,
                    {
                        "Update": {
                            "TableName": table_name,
                            "Key": {"user_id": user_id, "sk": customer["sk"]},
                            "UpdateExpression": customer_update,
                            "ConditionExpression": "attribute_exists(sk)",
                            "ExpressionAttributeValues": customer_values,
                        }
                    },
                    self._marker_put(
                        user_id, customer["phone"], customer_id, expire_at, now
                    ),
                ]
            )
        except ClientError as e:
            codes = _cancellation_codes(e)
            if len(codes) > 1 and codes[1] == "ConditionalCheckFailed":
                raise CustomerGoneError() from e
            if codes and codes[0] == "ConditionalCheckFailed":
                raise VisitChangedError() from e
            raise

        if created:
            return _plain(visit_item)
        return {**visit, "results": list(visit.get("results") or []) + [result]}

    def set_chosen_style(
        self, user_id: str, visit: Dict[str, Any], style: str
    ) -> Dict[str, Any]:
        """고른 스타일 저장

        Raises:
            VisitChangedError: 그 사이 상담 기록이 삭제됐다
        """
        try:
            self._call(
                self.table.update_item,
                Key={"user_id": user_id, "sk": visit["sk"]},
                UpdateExpression="SET chosen_style = :s",
                ConditionExpression="attribute_exists(sk)",
                ExpressionAttributeValues={":s": style},
            )
        except ClientError as e:
            if _error_code(e) == "ConditionalCheckFailedException":
                raise VisitChangedError() from e
            raise
        return {**visit, "chosen_style": style}


# Singleton
_pro_customer_repository: Optional[ProCustomerRepository] = None


def get_pro_customer_repository() -> ProCustomerRepository:
    global _pro_customer_repository
    if _pro_customer_repository is None:
        _pro_customer_repository = ProCustomerRepository()
    return _pro_customer_repository
