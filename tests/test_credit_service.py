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
