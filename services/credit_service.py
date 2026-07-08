"""크레딧 서비스 (DynamoDB 기반)

- 잔액: hairme-users 테이블의 credits 속성 (조건부 원자적 업데이트로 차감)
- 원장: hairme-credit-ledger 테이블에 모든 증감 내역 기록 (감사/CS용)
  - Partition Key: user_id
  - Sort Key: sk (ISO8601 타임스탬프 + 트랜잭션 ID, 시간순 정렬)
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

try:
    import boto3
    from botocore.exceptions import ClientError
    from botocore.config import Config

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from config.settings import settings
from core.logging import logger


class InsufficientCreditsError(Exception):
    """크레딧 잔액 부족"""

    def __init__(self, balance: int = 0):
        self.balance = balance
        super().__init__(f"크레딧이 부족합니다 (잔액: {balance})")


class CreditService:
    """크레딧 차감/지급/조회"""

    # 원장 reason 값: signup_bonus | synthesis | refund | purchase | admin_grant
    def __init__(self):
        self._users_table = None
        self._ledger_table = None

    def _resource(self):
        if not BOTO3_AVAILABLE:
            raise RuntimeError("boto3 is not installed")
        aws_region = os.getenv("AWS_REGION", settings.AWS_REGION)
        config = Config(connect_timeout=5, read_timeout=10, retries={"max_attempts": 3})
        return boto3.resource("dynamodb", region_name=aws_region, config=config)

    @property
    def users_table(self):
        if self._users_table is None:
            table_name = os.getenv(
                "DYNAMODB_USERS_TABLE_NAME", settings.DYNAMODB_USERS_TABLE_NAME
            )
            self._users_table = self._resource().Table(table_name)
        return self._users_table

    @property
    def ledger_table(self):
        if self._ledger_table is None:
            table_name = os.getenv(
                "DYNAMODB_CREDIT_LEDGER_TABLE_NAME",
                settings.DYNAMODB_CREDIT_LEDGER_TABLE_NAME,
            )
            self._ledger_table = self._resource().Table(table_name)
        return self._ledger_table

    def get_balance(self, user_id: str) -> int:
        """현재 잔액 조회"""
        try:
            response = self.users_table.get_item(
                Key={"user_id": user_id},
                ProjectionExpression="credits",
            )
        except ClientError as e:
            logger.error(f"잔액 조회 실패: {e.response['Error']['Message']}")
            raise

        item = response.get("Item")
        if item is None:
            raise ValueError("존재하지 않는 사용자입니다")
        return int(item.get("credits", 0))

    def consume(
        self,
        user_id: str,
        amount: int,
        reason: str = "synthesis",
        ref_id: Optional[str] = None,
    ) -> int:
        """
        크레딧 차감 (원자적 조건부 업데이트 - 잔액 부족 시 차감되지 않음)

        Returns:
            차감 후 잔액

        Raises:
            InsufficientCreditsError: 잔액 부족
        """
        if amount <= 0:
            raise ValueError("차감 금액은 1 이상이어야 합니다")

        try:
            response = self.users_table.update_item(
                Key={"user_id": user_id},
                UpdateExpression="SET credits = credits - :amt",
                ConditionExpression="attribute_exists(user_id) AND credits >= :amt",
                ExpressionAttributeValues={":amt": amount},
                ReturnValues="UPDATED_NEW",
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                balance = self._safe_balance(user_id)
                logger.info(f"크레딧 부족: user_id={user_id}, balance={balance}")
                raise InsufficientCreditsError(balance)
            logger.error(f"크레딧 차감 실패: {e.response['Error']['Message']}")
            raise

        balance_after = int(response["Attributes"]["credits"])
        self._write_ledger(user_id, -amount, reason, balance_after, ref_id)
        logger.info(
            f"크레딧 차감: user_id={user_id}, -{amount} ({reason}), 잔액={balance_after}"
        )
        return balance_after

    def grant(
        self,
        user_id: str,
        amount: int,
        reason: str,
        ref_id: Optional[str] = None,
    ) -> int:
        """
        크레딧 지급 (가입 보너스/구매/환불/관리자 지급)

        Returns:
            지급 후 잔액
        """
        if amount <= 0:
            raise ValueError("지급 금액은 1 이상이어야 합니다")

        try:
            response = self.users_table.update_item(
                Key={"user_id": user_id},
                UpdateExpression="SET credits = if_not_exists(credits, :zero) + :amt",
                ConditionExpression="attribute_exists(user_id)",
                ExpressionAttributeValues={":amt": amount, ":zero": 0},
                ReturnValues="UPDATED_NEW",
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise ValueError("존재하지 않는 사용자입니다")
            logger.error(f"크레딧 지급 실패: {e.response['Error']['Message']}")
            raise

        balance_after = int(response["Attributes"]["credits"])
        self._write_ledger(user_id, amount, reason, balance_after, ref_id)
        logger.info(
            f"크레딧 지급: user_id={user_id}, +{amount} ({reason}), 잔액={balance_after}"
        )
        return balance_after

    def get_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """최근 크레딧 증감 내역 (최신순)"""
        try:
            response = self.ledger_table.query(
                KeyConditionExpression="user_id = :uid",
                ExpressionAttributeValues={":uid": user_id},
                ScanIndexForward=False,
                Limit=limit,
            )
        except ClientError as e:
            logger.error(f"크레딧 내역 조회 실패: {e.response['Error']['Message']}")
            raise

        history = []
        for item in response.get("Items", []):
            history.append(
                {
                    "amount": int(item["amount"]),
                    "reason": item.get("reason"),
                    "balance_after": int(item.get("balance_after", 0)),
                    "created_at": item.get("created_at"),
                    "ref_id": item.get("ref_id"),
                }
            )
        return history

    def _safe_balance(self, user_id: str) -> int:
        try:
            return self.get_balance(user_id)
        except Exception:
            return 0

    def _write_ledger(
        self,
        user_id: str,
        amount: int,
        reason: str,
        balance_after: int,
        ref_id: Optional[str],
    ) -> None:
        """원장 기록 (실패해도 본 트랜잭션은 롤백하지 않음 - best effort)"""
        now = datetime.now(timezone.utc).isoformat()
        item: Dict[str, Any] = {
            "user_id": user_id,
            "sk": f"{now}#{uuid.uuid4().hex[:8]}",
            "amount": amount,
            "reason": reason,
            "balance_after": balance_after,
            "created_at": now,
        }
        if ref_id:
            item["ref_id"] = ref_id

        try:
            self.ledger_table.put_item(Item=item)
        except Exception as e:
            logger.error(f"⚠️ 크레딧 원장 기록 실패 (잔액은 반영됨): {str(e)}")


# Singleton
_credit_service: Optional[CreditService] = None


def get_credit_service() -> CreditService:
    global _credit_service
    if _credit_service is None:
        _credit_service = CreditService()
    return _credit_service
