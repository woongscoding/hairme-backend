"""크레딧 서비스 (DynamoDB 기반)

- 잔액: hairme-users 테이블의 credits 속성 (조건부 원자적 업데이트로 차감)
- 원장: hairme-credit-ledger 테이블에 모든 증감 내역 기록 (감사/CS용)
  - Partition Key: user_id
  - Sort Key: sk (ISO8601 타임스탬프 + 트랜잭션 ID, 시간순 정렬)
"""

import hashlib
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


def _mask_ref(ref_key: str) -> str:
    """ref_key의 토큰 부분을 해시로 마스킹 (네임스페이스 접두사는 유지)"""
    prefix, sep, secret = ref_key.partition("#")
    if not sep:
        prefix, secret = "", ref_key
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}#{digest}" if prefix else digest


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

    def try_claim_ref(
        self,
        ref_key: str,
        user_id: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        외부 트랜잭션(구매 토큰/광고 transaction_id) 중복 지급 방지 마커 (조건부 put)

        ledger 테이블에 user_id=ref_key(예: "purchase#<token>"), sk="claim" 아이템을
        attribute_not_exists 조건으로 기록한다. 사용자 원장(user_id=UUID)과
        키 공간이 겹치지 않아 같은 테이블을 재사용한다.

        Returns:
            True: 최초 처리 (지급 진행 가능)
            False: 이미 처리된 트랜잭션 (중복)
        """
        item: Dict[str, Any] = {
            "user_id": ref_key,
            "sk": "claim",
            "claimed_by": user_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if detail:
            item["detail"] = detail

        try:
            self.ledger_table.put_item(
                Item=item,
                ConditionExpression="attribute_not_exists(user_id)",
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                # ref_key에 구매 토큰 등이 포함되므로 원문 대신 해시로 로깅
                logger.warning(
                    f"⚠️ 중복 트랜잭션 처리 시도: ref_hash={_mask_ref(ref_key)}"
                )
                return False
            logger.error(f"트랜잭션 클레임 기록 실패: {e.response['Error']['Message']}")
            raise

    def release_ref(self, ref_key: str) -> None:
        """지급 실패 시 클레임 마커 회수 - 클라이언트 재시도 허용 (best effort)"""
        try:
            self.ledger_table.delete_item(Key={"user_id": ref_key, "sk": "claim"})
        except Exception as e:
            logger.error(f"⚠️ 클레임 마커 삭제 실패: {str(e)}")

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
        """원장 기록 (실패해도 본 트랜잭션은 롤백하지 않음)

        잔액은 이미 원자적으로 반영된 뒤라 여기서 실패하면 잔액↔원장이
        어긋난다. 1회 재시도하고, 최종 실패 시 CS 대사가 가능하도록
        복구에 필요한 전체 정보를 CRITICAL 구조화 로그로 남긴다.
        """
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

        for attempt in (1, 2):
            try:
                self.ledger_table.put_item(Item=item)
                return
            except Exception as e:
                if attempt == 1:
                    logger.warning(f"크레딧 원장 기록 1차 실패 - 재시도: {str(e)}")
                else:
                    # 수동 대사(재기록)에 필요한 모든 필드를 남긴다 (ref는 해시)
                    logger.critical(
                        "❌ LEDGER_WRITE_FAILED 크레딧 원장 기록 최종 실패 "
                        "(잔액은 반영됨 - 수동 대사 필요): "
                        f"user_id={user_id}, amount={amount}, reason={reason}, "
                        f"balance_after={balance_after}, created_at={now}, "
                        f"ref_hash={_mask_ref(ref_id) if ref_id else None}, "
                        f"error={str(e)}"
                    )


# Singleton
_credit_service: Optional[CreditService] = None


def get_credit_service() -> CreditService:
    global _credit_service
    if _credit_service is None:
        _credit_service = CreditService()
    return _credit_service
