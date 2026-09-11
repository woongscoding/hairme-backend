"""Play 환불 크레딧 회수 서비스 + 잡 디스패치 테스트 (HTTP/DynamoDB 모킹)"""

import json
import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")

from decimal import Decimal
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from botocore.exceptions import ClientError

from services.credit_service import CreditService
from services.play_billing_service import PlayBillingUnavailableError
from services.play_void_reclaim_service import (
    PlayVoidReclaimService,
    _mask_token,
)


def _voided(token: str, order_id: str = "GPA.0000"):
    return {
        "purchaseToken": token,
        "orderId": order_id,
        "voidedTimeMillis": "1720000000000",
        "voidedSource": 1,
        "voidedReason": 0,
    }


def _conditional_check_failed():
    return ClientError(
        error_response={
            "Error": {
                "Code": "ConditionalCheckFailedException",
                "Message": "The conditional request failed",
            }
        },
        operation_name="PutItem",
    )


def _claim_marker(user_id="user-1", product_id="credits_10"):
    return {
        "Item": {
            "user_id": "purchase#tok-1",
            "sk": "claim",
            "claimed_by": user_id,
            "detail": {"product_id": product_id, "order_id": "GPA.1111"},
        }
    }


@pytest.fixture
def play():
    """Play API 모킹 (package_name 설정됨 + 자격증명 정상)"""
    mock = MagicMock()
    mock.package_name = "com.hairme.app"
    return mock


@pytest.fixture
def credits():
    """실제 CreditService + DynamoDB 테이블만 모킹"""
    svc = CreditService()
    svc._users_table = MagicMock()
    svc._ledger_table = MagicMock()
    return svc


def _ledger_entries(credits_service):
    """put_item 호출 중 원장(=claim 마커가 아닌) 항목만 추출"""
    return [
        call.kwargs["Item"]
        for call in credits_service._ledger_table.put_item.call_args_list
        if call.kwargs["Item"].get("sk") != "claim"
    ]


def _claim_items(credits_service):
    return [
        call.kwargs["Item"]
        for call in credits_service._ledger_table.put_item.call_args_list
        if call.kwargs["Item"].get("sk") == "claim"
    ]


class TestDisabled:
    def test_disabled_when_package_name_missing(self, credits):
        play = MagicMock()
        play.package_name = ""

        summary = PlayVoidReclaimService(play, credits).run()

        assert summary["disabled"] is True
        assert summary["reason"] == "package_name_not_configured"
        play.list_voided_purchases.assert_not_called()

    def test_disabled_when_credentials_unavailable(self, credits):
        play = MagicMock()
        play.package_name = "com.hairme.app"
        type(play).session = PropertyMock(
            side_effect=PlayBillingUnavailableError("service account key missing")
        )

        summary = PlayVoidReclaimService(play, credits).run()

        assert summary["disabled"] is True
        assert summary["reason"] == "credentials_unavailable"
        play.list_voided_purchases.assert_not_called()


class TestPagination:
    def test_follows_next_page_token(self, play, credits):
        play.list_voided_purchases.side_effect = [
            {
                "voidedPurchases": [_voided("tok-a")],
                "tokenPagination": {"nextPageToken": "page-2"},
            },
            {"voidedPurchases": [_voided("tok-b")]},
        ]
        credits._ledger_table.get_item.return_value = {}  # 마커 없음 → not_found

        summary = PlayVoidReclaimService(play, credits).run(lookback_days=7)

        assert summary["checked"] == 2
        assert summary["not_found"] == 2
        assert play.list_voided_purchases.call_count == 2
        assert play.list_voided_purchases.call_args_list[0].kwargs["page_token"] is None
        assert (
            play.list_voided_purchases.call_args_list[1].kwargs["page_token"]
            == "page-2"
        )
        # startTime 은 lookback_days 만큼 과거 (epoch ms)
        start_ms = play.list_voided_purchases.call_args_list[0].args[0]
        assert isinstance(start_ms, int) and start_ms > 0

    def test_stops_when_no_pagination(self, play, credits):
        play.list_voided_purchases.return_value = {"voidedPurchases": []}

        summary = PlayVoidReclaimService(play, credits).run()

        assert play.list_voided_purchases.call_count == 1
        assert summary["checked"] == 0


class TestReclaim:
    def test_reclaim_writes_marker_and_ledger_and_allows_negative(self, play, credits):
        play.list_voided_purchases.return_value = {
            "voidedPurchases": [_voided("tok-1", "GPA.1111")]
        }
        credits._ledger_table.get_item.return_value = _claim_marker()
        # 이미 다 써버린 사용자 → 회수 후 음수
        credits._users_table.update_item.return_value = {
            "Attributes": {"credits": Decimal("-7")}
        }

        summary = PlayVoidReclaimService(play, credits).run()

        assert summary["reclaimed"] == 1
        assert summary["checked"] == 1
        assert summary["failed"] == 0

        # 잔액 차감에 잔액 조건이 없어야 함 (음수 허용)
        update_kwargs = credits._users_table.update_item.call_args.kwargs
        assert "credits >=" not in update_kwargs["ConditionExpression"]
        assert update_kwargs["ExpressionAttributeValues"][":amt"] == 10

        # 회수 멱등 마커
        claims = _claim_items(credits)
        assert len(claims) == 1
        assert claims[0]["user_id"] == "void#tok-1"
        assert claims[0]["claimed_by"] == "user-1"

        # 원장 기록
        entries = _ledger_entries(credits)
        assert len(entries) == 1
        assert entries[0]["amount"] == -10
        assert entries[0]["reason"] == "void_reclaim"
        assert entries[0]["balance_after"] == -7
        assert entries[0]["ref_id"] == "tok-1"

    def test_amount_falls_back_to_ledger_when_product_unknown(self, play, credits):
        play.list_voided_purchases.return_value = {
            "voidedPurchases": [_voided("tok-1")]
        }
        credits._ledger_table.get_item.return_value = _claim_marker(
            product_id="credits_legacy_7"
        )
        credits._ledger_table.query.return_value = {
            "Items": [
                {
                    "amount": Decimal("7"),
                    "reason": "purchase",
                    "ref_id": "tok-1",
                }
            ]
        }
        credits._users_table.update_item.return_value = {
            "Attributes": {"credits": Decimal("0")}
        }

        summary = PlayVoidReclaimService(play, credits).run()

        assert summary["reclaimed"] == 1
        assert (
            credits._users_table.update_item.call_args.kwargs[
                "ExpressionAttributeValues"
            ][":amt"]
            == 7
        )

    def test_second_run_is_idempotent(self, play, credits):
        play.list_voided_purchases.return_value = {
            "voidedPurchases": [_voided("tok-1")]
        }
        credits._ledger_table.get_item.return_value = _claim_marker()
        # void# 마커가 이미 존재 → 조건부 put 실패
        credits._ledger_table.put_item.side_effect = _conditional_check_failed()

        summary = PlayVoidReclaimService(play, credits).run()

        assert summary["already_reclaimed"] == 1
        assert summary["reclaimed"] == 0
        credits._users_table.update_item.assert_not_called()

    def test_unknown_token_is_not_found(self, play, credits):
        play.list_voided_purchases.return_value = {
            "voidedPurchases": [_voided("tok-unknown")]
        }
        credits._ledger_table.get_item.return_value = {}

        summary = PlayVoidReclaimService(play, credits).run()

        assert summary["not_found"] == 1
        assert summary["reclaimed"] == 0
        credits._users_table.update_item.assert_not_called()

    def test_deleted_user_counts_as_not_found_and_keeps_marker(self, play, credits):
        play.list_voided_purchases.return_value = {
            "voidedPurchases": [_voided("tok-1")]
        }
        credits._ledger_table.get_item.return_value = _claim_marker()
        credits._users_table.update_item.side_effect = ClientError(
            error_response={
                "Error": {
                    "Code": "ConditionalCheckFailedException",
                    "Message": "no user",
                }
            },
            operation_name="UpdateItem",
        )

        summary = PlayVoidReclaimService(play, credits).run()

        assert summary["not_found"] == 1
        assert summary["failed"] == 0
        credits._ledger_table.delete_item.assert_not_called()

    def test_reclaim_failure_releases_marker_and_counts_failed(self, play, credits):
        play.list_voided_purchases.return_value = {
            "voidedPurchases": [_voided("tok-1")]
        }
        credits._ledger_table.get_item.return_value = _claim_marker()
        credits._users_table.update_item.side_effect = RuntimeError("dynamo down")

        summary = PlayVoidReclaimService(play, credits).run()

        assert summary["failed"] == 1
        assert summary["reclaimed"] == 0
        credits._ledger_table.delete_item.assert_called_once()


class TestApiFailure:
    @pytest.mark.parametrize("status", [401, 500])
    def test_api_error_counts_failed_without_raising(self, play, credits, status):
        play.list_voided_purchases.side_effect = PlayBillingUnavailableError(
            f"voidedpurchases api error: {status}"
        )

        summary = PlayVoidReclaimService(play, credits).run()

        assert summary["failed"] == 1
        assert summary["checked"] == 0
        assert summary["disabled"] is False

    def test_entry_error_does_not_abort_remaining(self, play, credits):
        play.list_voided_purchases.return_value = {
            "voidedPurchases": [_voided("tok-bad"), _voided("tok-ok")]
        }
        credits._ledger_table.get_item.side_effect = [
            RuntimeError("dynamo down"),
            _claim_marker(),
        ]
        credits._users_table.update_item.return_value = {
            "Attributes": {"credits": Decimal("1")}
        }

        summary = PlayVoidReclaimService(play, credits).run()

        assert summary["failed"] == 1
        assert summary["reclaimed"] == 1


class TestTokenMasking:
    def test_structured_log_masks_token(self, play, credits):
        play.list_voided_purchases.return_value = {
            "voidedPurchases": [_voided("super-secret-token")]
        }
        credits._ledger_table.get_item.return_value = {}

        with patch("services.play_void_reclaim_service.log_structured") as mock_log:
            PlayVoidReclaimService(play, credits).run()

        event_type, payload = mock_log.call_args.args
        assert event_type == "play_void_reclaim"
        assert payload["token_hash"] == _mask_token("super-secret-token")
        assert "super-secret-token" not in json.dumps(payload)


class TestJobDispatch:
    def test_job_event_runs_service(self):
        import main

        service = MagicMock()
        service.run.return_value = {
            "checked": 3,
            "reclaimed": 1,
            "already_reclaimed": 1,
            "not_found": 1,
            "failed": 0,
            "disabled": False,
        }

        mangum = MagicMock()
        with patch.object(main, "_mangum_handler", mangum), patch(
            "services.play_void_reclaim_service.get_play_void_reclaim_service",
            return_value=service,
        ):
            response = main.handler({"job": "reclaim_voided_purchases"}, None)

        assert response["statusCode"] == 200
        assert json.loads(response["body"])["reclaimed"] == 1
        service.run.assert_called_once()
        mangum.assert_not_called()

    def test_job_respects_disabled_setting(self, monkeypatch):
        import main

        monkeypatch.setattr(main.settings, "PLAY_VOID_RECLAIM_ENABLED", False)

        with patch(
            "services.play_void_reclaim_service.get_play_void_reclaim_service"
        ) as factory:
            response = main.handler({"job": "reclaim_voided_purchases"}, None)

        assert response == {"statusCode": 200, "body": "disabled"}
        factory.assert_not_called()

    def test_unknown_job_returns_400(self):
        import main

        mangum = MagicMock()
        with patch.object(main, "_mangum_handler", mangum):
            response = main.handler({"job": "no_such_job"}, None)

        assert response["statusCode"] == 400
        assert json.loads(response["body"])["error"] == "unknown job"
        mangum.assert_not_called()

    def test_job_exception_returns_500(self):
        import main

        with patch(
            "services.play_void_reclaim_service.get_play_void_reclaim_service",
            side_effect=RuntimeError("boom"),
        ):
            response = main.handler({"job": "reclaim_voided_purchases"}, None)

        assert response["statusCode"] == 500
        assert json.loads(response["body"])["error"] == "job failed"

    def test_warmup_event_still_works(self):
        import main

        with patch.object(main, "warm_up") as mock_warm:
            response = main.handler({"warmup": True}, None)

        assert response == {"statusCode": 200, "body": "warm"}
        mock_warm.assert_called_once()
