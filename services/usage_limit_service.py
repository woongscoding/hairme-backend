"""Daily usage limit service using DynamoDB (HairstyleDailyUsage table)"""

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

try:
    import boto3
    from botocore.exceptions import ClientError
    from botocore.config import Config

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from config.settings import settings
from core.logging import logger

# KST timezone (UTC+9)
KST = timezone(timedelta(hours=9))

# device_id 형식 제한
# - 이 테이블의 파티션 키는 "reward_ad#<user_id>", "ip#<addr>" 같은 서버 전용
#   네임스페이스 키와 공유된다. 클라이언트가 보낸 device_id에 '#'이 들어가면
#   타인의 리워드/IP 카운터를 조작할 수 있으므로 '#'과 공백을 금지한다.
# - Android ID(16자리 hex), UUID(하이픈 포함), 일반적인 설치 ID는 모두 통과한다.
DEVICE_ID_MIN_LENGTH = 8
DEVICE_ID_MAX_LENGTH = 128
DEVICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]+$")


def validate_device_id(device_id: str) -> str:
    """
    device_id 형식 검증 후 정규화(strip)된 값 반환.

    Raises:
        ValueError: 형식이 올바르지 않은 경우 (엔드포인트에서 400으로 매핑)
    """
    if not isinstance(device_id, str):
        raise ValueError("device_id는 필수입니다.")

    trimmed = device_id.strip()
    if not trimmed:
        raise ValueError("device_id는 필수입니다.")

    if not (DEVICE_ID_MIN_LENGTH <= len(trimmed) <= DEVICE_ID_MAX_LENGTH):
        raise ValueError(
            f"device_id는 {DEVICE_ID_MIN_LENGTH}~{DEVICE_ID_MAX_LENGTH}자여야 합니다."
        )

    if not DEVICE_ID_PATTERN.match(trimmed):
        raise ValueError("device_id 형식이 올바르지 않습니다.")

    return trimmed


class UsageLimitService:
    """
    Service for managing daily synthesis usage limits per device.

    Uses DynamoDB table 'hairstyle_usage' with:
    - Partition Key: device_id (String)
    - Sort Key: date (String, YYYY-MM-DD in KST)
    - Attributes: count (Number), expire_at (Number, epoch seconds for TTL)
    """

    def __init__(self):
        self._table = None

    @property
    def table(self):
        """Lazy load the DynamoDB table resource"""
        if self._table is None:
            if not BOTO3_AVAILABLE:
                raise RuntimeError("boto3 is not installed")

            aws_region = os.getenv("AWS_REGION", "ap-northeast-2")
            table_name = os.getenv(
                "DYNAMODB_USAGE_TABLE_NAME",
                settings.DYNAMODB_USAGE_TABLE_NAME,
            )
            config = Config(
                connect_timeout=5, read_timeout=10, retries={"max_attempts": 3}
            )
            resource = boto3.resource("dynamodb", region_name=aws_region, config=config)
            self._table = resource.Table(table_name)
        return self._table

    @property
    def daily_limit(self) -> int:
        return settings.DAILY_SYNTHESIS_LIMIT

    @staticmethod
    def _today_kst() -> str:
        """Get today's date string in KST (YYYY-MM-DD)"""
        return datetime.now(KST).strftime("%Y-%m-%d")

    @staticmethod
    def _tomorrow_kst_epoch() -> int:
        """Get epoch timestamp for midnight tomorrow KST (for TTL)"""
        now_kst = datetime.now(KST)
        tomorrow = (now_kst + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return int(tomorrow.timestamp())

    def check_usage(self, device_id: str) -> Dict[str, Any]:
        """
        Check if the device has remaining usage (read-only, no increment).

        Args:
            device_id: Unique device identifier

        Returns:
            {
                "allowed": bool,
                "daily_limit": int,
                "used": int,
                "remaining": int,
            }
        """
        usage = self.get_usage(validate_device_id(device_id))
        allowed = usage["used"] < self.daily_limit
        return {
            "allowed": allowed,
            "daily_limit": usage["daily_limit"],
            "used": usage["used"],
            "remaining": usage["remaining"],
        }

    def increment_usage(self, device_id: str) -> Dict[str, Any]:
        """
        Atomically increment usage count for a device.
        Call this AFTER a successful synthesis.

        Uses DynamoDB conditional UpdateExpression to prevent race conditions.

        Args:
            device_id: Unique device identifier

        Returns:
            { "daily_limit": int, "used": int, "remaining": int }

        Raises:
            ValueError: device_id 형식이 올바르지 않은 경우
        """
        device_id = validate_device_id(device_id)
        today = self._today_kst()
        ttl_value = self._tomorrow_kst_epoch()

        try:
            response = self.table.update_item(
                Key={"device_id": device_id, "date": today},
                UpdateExpression="SET #cnt = if_not_exists(#cnt, :zero) + :inc, #ttl = :ttl",
                ConditionExpression="attribute_not_exists(#cnt) OR #cnt < :limit",
                ExpressionAttributeNames={
                    "#cnt": "count",
                    "#ttl": "expire_at",
                },
                ExpressionAttributeValues={
                    ":zero": 0,
                    ":inc": 1,
                    ":limit": self.daily_limit,
                    ":ttl": ttl_value,
                },
                ReturnValues="ALL_NEW",
            )

            used = int(response["Attributes"]["count"])
            remaining = max(0, self.daily_limit - used)

            logger.info(
                f"Usage incremented: device={device_id}, date={today}, "
                f"used={used}/{self.daily_limit}"
            )

            return {
                "daily_limit": self.daily_limit,
                "used": used,
                "remaining": remaining,
            }

        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.warning(
                    f"Increment failed (limit reached): device={device_id}, date={today}"
                )
                return {
                    "daily_limit": self.daily_limit,
                    "used": self.daily_limit,
                    "remaining": 0,
                }
            logger.error(
                f"DynamoDB usage update failed: {e.response['Error']['Message']}"
            )
            raise

    def check_and_increment_usage(self, device_id: str) -> Dict[str, Any]:
        """
        Atomically check and increment usage count for a device.
        Kept for backward compatibility. Prefer check_usage() + increment_usage().

        Raises:
            ValueError: device_id 형식이 올바르지 않은 경우
        """
        device_id = validate_device_id(device_id)
        today = self._today_kst()
        ttl_value = self._tomorrow_kst_epoch()

        try:
            response = self.table.update_item(
                Key={"device_id": device_id, "date": today},
                UpdateExpression="SET #cnt = if_not_exists(#cnt, :zero) + :inc, #ttl = :ttl",
                ConditionExpression="attribute_not_exists(#cnt) OR #cnt < :limit",
                ExpressionAttributeNames={
                    "#cnt": "count",
                    "#ttl": "expire_at",
                },
                ExpressionAttributeValues={
                    ":zero": 0,
                    ":inc": 1,
                    ":limit": self.daily_limit,
                    ":ttl": ttl_value,
                },
                ReturnValues="ALL_NEW",
            )

            used = int(response["Attributes"]["count"])
            remaining = max(0, self.daily_limit - used)

            logger.info(
                f"Usage incremented: device={device_id}, date={today}, "
                f"used={used}/{self.daily_limit}"
            )

            return {
                "allowed": True,
                "daily_limit": self.daily_limit,
                "used": used,
                "remaining": remaining,
            }

        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.info(f"Daily limit reached: device={device_id}, date={today}")
                return {
                    "allowed": False,
                    "daily_limit": self.daily_limit,
                    "used": self.daily_limit,
                    "remaining": 0,
                }
            logger.error(
                f"DynamoDB usage update failed: {e.response['Error']['Message']}"
            )
            raise

    def increment_daily_counter(self, key: str, limit: int) -> bool:
        """
        범용 일일 카운터 (KST 기준, 조건부 원자 증가, TTL 자동 만료)

        synthesis 외 용도의 일일 상한에도 같은 테이블을 재사용한다.
        키는 "reward_ad#{user_id}" 같은 네임스페이스로 구분할 것.

        Returns:
            True: 증가 성공 (상한 미달), False: 일일 상한 도달
        """
        today = self._today_kst()

        try:
            self.table.update_item(
                Key={"device_id": key, "date": today},
                UpdateExpression="SET #cnt = if_not_exists(#cnt, :zero) + :inc, #ttl = :ttl",
                ConditionExpression="attribute_not_exists(#cnt) OR #cnt < :limit",
                ExpressionAttributeNames={
                    "#cnt": "count",
                    "#ttl": "expire_at",
                },
                ExpressionAttributeValues={
                    ":zero": 0,
                    ":inc": 1,
                    ":limit": limit,
                    ":ttl": self._tomorrow_kst_epoch(),
                },
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.info(f"일일 카운터 상한 도달: key={key}, date={today}")
                return False
            logger.error(
                f"DynamoDB 일일 카운터 업데이트 실패: {e.response['Error']['Message']}"
            )
            raise

    def decrement_daily_counter(self, key: str) -> None:
        """
        일일 카운터 되돌리기 (best effort)

        카운터 증가 후 후속 처리(크레딧 지급 등)가 실패했을 때 호출한다.
        복구가 실패해도 예외를 전파하지 않는다 - 사용자 한도가 1 소모되는
        것이 지급 흐름 전체가 깨지는 것보다 낫기 때문.
        """
        today = self._today_kst()

        try:
            self.table.update_item(
                Key={"device_id": key, "date": today},
                UpdateExpression="SET #cnt = #cnt - :dec",
                ConditionExpression="attribute_exists(#cnt) AND #cnt > :zero",
                ExpressionAttributeNames={"#cnt": "count"},
                ExpressionAttributeValues={":dec": 1, ":zero": 0},
            )
            logger.info(f"일일 카운터 복구: key={key}, date={today}")
        except Exception as e:
            logger.error(f"⚠️ 일일 카운터 복구 실패 (무시): key={key}, {str(e)}")

    def decrement_usage(self, device_id: str) -> None:
        """
        디바이스 일일 사용량 되돌리기 (best effort)

        합성이 실패해 사용자에게 결과가 전달되지 않았을 때 무료 한도를 복구한다.
        복구 실패는 전파하지 않는다 (decrement_daily_counter와 동일한 정책).
        """
        try:
            validated = validate_device_id(device_id)
        except ValueError as e:
            logger.error(f"⚠️ 사용량 복구 스킵 (잘못된 device_id): {str(e)}")
            return
        self.decrement_daily_counter(validated)

    def get_usage(self, device_id: str) -> Dict[str, Any]:
        """
        Get current usage info for a device.

        Args:
            device_id: Unique device identifier

        Returns:
            { "daily_limit": int, "used": int, "remaining": int }

        Raises:
            ValueError: device_id 형식이 올바르지 않은 경우
        """
        device_id = validate_device_id(device_id)
        today = self._today_kst()

        try:
            response = self.table.get_item(Key={"device_id": device_id, "date": today})

            if "Item" not in response:
                return {
                    "daily_limit": self.daily_limit,
                    "used": 0,
                    "remaining": self.daily_limit,
                }

            used = int(response["Item"].get("count", 0))
            remaining = max(0, self.daily_limit - used)

            return {
                "daily_limit": self.daily_limit,
                "used": used,
                "remaining": remaining,
            }

        except ClientError as e:
            logger.error(
                f"DynamoDB usage query failed: {e.response['Error']['Message']}"
            )
            raise


# Singleton
_usage_limit_service: Optional[UsageLimitService] = None


def get_usage_limit_service() -> UsageLimitService:
    """Get or create the usage limit service singleton"""
    global _usage_limit_service
    if _usage_limit_service is None:
        _usage_limit_service = UsageLimitService()
    return _usage_limit_service
