"""Daily usage limit service using DynamoDB (HairstyleDailyUsage table)"""

import os
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
        usage = self.get_usage(device_id)
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
        """
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
        """
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

    def get_usage(self, device_id: str) -> Dict[str, Any]:
        """
        Get current usage info for a device.

        Args:
            device_id: Unique device identifier

        Returns:
            { "daily_limit": int, "used": int, "remaining": int }
        """
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
