"""크레딧 서비스 테스트 (DynamoDB 모킹)"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")

from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from services.credit_service import CreditService, InsufficientCreditsError


def _conditional_check_failed():
    return ClientError(
        error_response={
            "Error": {
                "Code": "ConditionalCheckFailedException",
                "Message": "The conditional request failed",
            }
        },
        operation_name="UpdateItem",
    )


@pytest.fixture
def service():
    svc = CreditService()
    svc._users_table = MagicMock()
    svc._ledger_table = MagicMock()
    return svc


class TestConsume:
    def test_consume_success(self, service):
        service._users_table.update_item.return_value = {
            "Attributes": {"credits": Decimal("4")}
        }

        balance = service.consume("user-1", 1, reason="synthesis")

        assert balance == 4
        # 조건부 차감인지 확인 (잔액 부족 시 차감 방지)
        call_kwargs = service._users_table.update_item.call_args.kwargs
        assert "credits >= :amt" in call_kwargs["ConditionExpression"]
        # 원장 기록 확인
        ledger_item = service._ledger_table.put_item.call_args.kwargs["Item"]
        assert ledger_item["amount"] == -1
        assert ledger_item["reason"] == "synthesis"
        assert ledger_item["balance_after"] == 4

    def test_consume_insufficient_credits(self, service):
        service._users_table.update_item.side_effect = _conditional_check_failed()
        service._users_table.get_item.return_value = {"Item": {"credits": Decimal("0")}}

        with pytest.raises(InsufficientCreditsError) as exc_info:
            service.consume("user-1", 1)

        assert exc_info.value.balance == 0
        # 실패 시 원장 기록 없음
        service._ledger_table.put_item.assert_not_called()

    def test_consume_invalid_amount(self, service):
        with pytest.raises(ValueError):
            service.consume("user-1", 0)
        with pytest.raises(ValueError):
            service.consume("user-1", -5)


class TestGrant:
    def test_grant_success(self, service):
        service._users_table.update_item.return_value = {
            "Attributes": {"credits": Decimal("15")}
        }

        balance = service.grant("user-1", 10, reason="purchase", ref_id="order-123")

        assert balance == 15
        ledger_item = service._ledger_table.put_item.call_args.kwargs["Item"]
        assert ledger_item["amount"] == 10
        assert ledger_item["reason"] == "purchase"
        assert ledger_item["ref_id"] == "order-123"

    def test_grant_to_nonexistent_user(self, service):
        service._users_table.update_item.side_effect = _conditional_check_failed()

        with pytest.raises(ValueError):
            service.grant("ghost-user", 10, reason="admin_grant")

    def test_ledger_failure_does_not_break_grant(self, service):
        """원장 기록 실패해도 지급 자체는 성공 (best effort)"""
        service._users_table.update_item.return_value = {
            "Attributes": {"credits": Decimal("5")}
        }
        service._ledger_table.put_item.side_effect = Exception("DynamoDB down")

        balance = service.grant("user-1", 5, reason="signup_bonus")
        assert balance == 5


class TestClaimRef:
    def test_first_claim_succeeds(self, service):
        result = service.try_claim_ref("purchase#token-1", "user-1")

        assert result is True
        call_kwargs = service._ledger_table.put_item.call_args.kwargs
        # 조건부 put으로 중복 방지하는지 확인
        assert "attribute_not_exists" in call_kwargs["ConditionExpression"]
        assert call_kwargs["Item"]["user_id"] == "purchase#token-1"
        assert call_kwargs["Item"]["claimed_by"] == "user-1"

    def test_duplicate_claim_returns_false(self, service):
        service._ledger_table.put_item.side_effect = ClientError(
            error_response={
                "Error": {
                    "Code": "ConditionalCheckFailedException",
                    "Message": "The conditional request failed",
                }
            },
            operation_name="PutItem",
        )

        assert service.try_claim_ref("purchase#token-1", "user-2") is False

    def test_release_ref_deletes_marker(self, service):
        service.release_ref("purchase#token-1")

        service._ledger_table.delete_item.assert_called_once_with(
            Key={"user_id": "purchase#token-1", "sk": "claim"}
        )

    def test_release_ref_swallows_errors(self, service):
        """마커 삭제 실패는 best effort - 예외를 전파하지 않음"""
        service._ledger_table.delete_item.side_effect = Exception("DynamoDB down")
        service.release_ref("purchase#token-1")  # 예외 없이 통과


class TestBalanceAndHistory:
    def test_get_balance(self, service):
        service._users_table.get_item.return_value = {"Item": {"credits": Decimal("7")}}
        assert service.get_balance("user-1") == 7

    def test_get_balance_nonexistent_user(self, service):
        service._users_table.get_item.return_value = {}
        with pytest.raises(ValueError):
            service.get_balance("ghost-user")

    def test_get_history(self, service):
        service._ledger_table.query.return_value = {
            "Items": [
                {
                    "amount": Decimal("-1"),
                    "reason": "synthesis",
                    "balance_after": Decimal("4"),
                    "created_at": "2026-07-06T00:00:00+00:00",
                },
                {
                    "amount": Decimal("5"),
                    "reason": "signup_bonus",
                    "balance_after": Decimal("5"),
                    "created_at": "2026-07-05T00:00:00+00:00",
                },
            ]
        }

        history = service.get_history("user-1")

        assert len(history) == 2
        assert history[0]["amount"] == -1
        assert history[1]["reason"] == "signup_bonus"
        # 최신순 조회인지 확인
        call_kwargs = service._ledger_table.query.call_args.kwargs
        assert call_kwargs["ScanIndexForward"] is False


class TestConsumeForReclaim:
    """환불 회수 차감 - 잔액이 음수가 되는 것을 허용"""

    def test_allows_negative_balance(self, service):
        service._users_table.update_item.return_value = {
            "Attributes": {"credits": Decimal("-7")}
        }

        balance = service.consume_for_reclaim("user-1", 10, ref_id="tok-1")

        assert balance == -7
        call_kwargs = service._users_table.update_item.call_args.kwargs
        # 잔액 조건이 없어야 한다 (일반 consume 과의 차이)
        assert "credits >=" not in call_kwargs["ConditionExpression"]
        assert "attribute_exists(user_id)" in call_kwargs["ConditionExpression"]

        ledger_item = service._ledger_table.put_item.call_args.kwargs["Item"]
        assert ledger_item["amount"] == -10
        assert ledger_item["reason"] == "void_reclaim"
        assert ledger_item["balance_after"] == -7
        assert ledger_item["ref_id"] == "tok-1"

    def test_negative_balance_blocks_further_consume(self, service):
        """회수로 음수가 된 뒤에는 조건식(credits >= :amt)에 막혀 합성 불가"""
        service._users_table.update_item.side_effect = _conditional_check_failed()
        service._users_table.get_item.return_value = {
            "Item": {"credits": Decimal("-7")}
        }

        with pytest.raises(InsufficientCreditsError) as exc_info:
            service.consume("user-1", 1)

        assert exc_info.value.balance == -7

    def test_missing_user_raises_value_error(self, service):
        service._users_table.update_item.side_effect = _conditional_check_failed()

        with pytest.raises(ValueError):
            service.consume_for_reclaim("user-1", 10)

        service._ledger_table.put_item.assert_not_called()

    def test_invalid_amount(self, service):
        with pytest.raises(ValueError):
            service.consume_for_reclaim("user-1", 0)


class TestClaimLookup:
    """구매 토큰 → 사용자/금액 역추적 (환불 회수용)"""

    def test_get_claim_returns_marker(self, service):
        service._ledger_table.get_item.return_value = {
            "Item": {"claimed_by": "user-1", "detail": {"product_id": "credits_10"}}
        }

        marker = service.get_claim("purchase#tok-1")

        assert marker["claimed_by"] == "user-1"
        key = service._ledger_table.get_item.call_args.kwargs["Key"]
        assert key == {"user_id": "purchase#tok-1", "sk": "claim"}

    def test_get_claim_returns_none_when_missing(self, service):
        service._ledger_table.get_item.return_value = {}

        assert service.get_claim("purchase#unknown") is None

    def test_find_ledger_entry_filters_by_ref_and_reason(self, service):
        service._ledger_table.query.return_value = {
            "Items": [{"amount": Decimal("10"), "ref_id": "tok-1"}]
        }

        entry = service.find_ledger_entry("user-1", "tok-1", reason="purchase")

        assert entry["amount"] == Decimal("10")
        call_kwargs = service._ledger_table.query.call_args.kwargs
        assert call_kwargs["ExpressionAttributeValues"][":ref"] == "tok-1"
        assert call_kwargs["ExpressionAttributeValues"][":reason"] == "purchase"
        # 예약어 충돌 방지를 위해 이름 치환 사용
        assert call_kwargs["ExpressionAttributeNames"]["#reason"] == "reason"

    def test_find_ledger_entry_returns_none_when_empty(self, service):
        service._ledger_table.query.return_value = {"Items": []}

        assert service.find_ledger_entry("user-1", "tok-x", reason="purchase") is None
