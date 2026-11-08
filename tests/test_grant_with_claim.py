"""CreditService.grant_with_claim - 마커·지급·원장 원자성 (moto 가짜 DynamoDB)

예전 구매 흐름은 try_claim_ref → grant 를 따로 해서, 그 사이에 Lambda 가 끊기면
마커만 남고 크레딧은 지급되지 않았다 (재시도는 409 로 막힘). 여기서는 셋이
한 트랜잭션으로 묶여 "전부 반영 또는 전부 미반영"인지 확인한다.
"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")

from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

from services.credit_service import CreditService, RefAlreadyClaimedError

pytestmark = pytest.mark.moto

REGION = "ap-northeast-2"
USER_ID = "buyer-0001"
REF_KEY = "purchase#token-1"


@pytest.fixture
def ddb(monkeypatch):
    monkeypatch.setenv("AWS_REGION", REGION)
    with mock_aws():
        resource = boto3.resource("dynamodb", region_name=REGION)
        resource.create_table(
            TableName="hairme-users",
            KeySchema=[{"AttributeName": "user_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "user_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        resource.create_table(
            TableName="hairme-credit-ledger",
            KeySchema=[
                {"AttributeName": "user_id", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "user_id", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        resource.Table("hairme-users").put_item(
            Item={"user_id": USER_ID, "credits": 3, "paid_credits": 0}
        )
        yield resource


def _ledger_rows(ddb, user_id=USER_ID):
    return ddb.Table("hairme-credit-ledger").query(
        KeyConditionExpression="user_id = :u",
        ExpressionAttributeValues={":u": user_id},
    )["Items"]


def _claim(ddb, ref_key=REF_KEY):
    return (
        ddb.Table("hairme-credit-ledger")
        .get_item(Key={"user_id": ref_key, "sk": "claim"})
        .get("Item")
    )


def test_grants_and_records_claim_and_ledger(ddb):
    service = CreditService()

    balance = service.grant_with_claim(
        REF_KEY,
        USER_ID,
        10,
        reason="purchase",
        ref_id="token-1",
        paid=True,
        detail={"product_id": "credits_10", "order_id": "GPA.1"},
    )

    assert balance == 13
    user = ddb.Table("hairme-users").get_item(Key={"user_id": USER_ID})["Item"]
    assert int(user["credits"]) == 13
    assert int(user["paid_credits"]) == 10
    assert user.get("first_purchase_at")

    claim = _claim(ddb)
    assert claim["claimed_by"] == USER_ID
    assert claim["detail"]["product_id"] == "credits_10"

    rows = _ledger_rows(ddb)
    assert len(rows) == 1
    assert int(rows[0]["amount"]) == 10
    assert int(rows[0]["balance_after"]) == 13
    assert rows[0]["reason"] == "purchase"
    assert rows[0]["ref_id"] == "token-1"

    # 환불 회수 잡이 쓰는 역추적 경로가 그대로 동작
    assert service.get_claim(REF_KEY)["claimed_by"] == USER_ID
    assert service.find_ledger_entry(USER_ID, "token-1", "purchase") is not None


def test_duplicate_ref_raises_and_grants_nothing(ddb):
    service = CreditService()
    service.grant_with_claim(REF_KEY, USER_ID, 10, reason="purchase", paid=True)

    with pytest.raises(RefAlreadyClaimedError):
        service.grant_with_claim(REF_KEY, USER_ID, 10, reason="purchase", paid=True)

    user = ddb.Table("hairme-users").get_item(Key={"user_id": USER_ID})["Item"]
    assert int(user["credits"]) == 13
    assert len(_ledger_rows(ddb)) == 1


def test_unknown_user_raises_value_error_without_claim(ddb):
    service = CreditService()

    with pytest.raises(ValueError):
        service.grant_with_claim(REF_KEY, "nobody", 10, reason="purchase")

    assert _claim(ddb) is None


def test_user_without_credits_attribute(ddb):
    ddb.Table("hairme-users").put_item(Item={"user_id": "fresh"})
    service = CreditService()

    balance = service.grant_with_claim("purchase#t2", "fresh", 5, reason="purchase")

    assert balance == 5
    assert int(_ledger_rows(ddb, "fresh")[0]["balance_after"]) == 5


def test_concurrent_balance_change_retries_with_exact_balance(ddb):
    """읽은 뒤 다른 요청이 잔액을 바꾸면 조건 실패 → 다시 읽어 정확한 잔액으로 지급"""
    service = CreditService()
    users = ddb.Table("hairme-users")
    original_get = users.get_item
    calls = {"n": 0}

    def racing_get_item(**kwargs):
        result = original_get(**kwargs)
        calls["n"] += 1
        if calls["n"] == 1:
            # 첫 조회 직후 동시 합성이 1 크레딧 차감
            users.update_item(
                Key={"user_id": USER_ID},
                UpdateExpression="SET credits = credits - :one",
                ExpressionAttributeValues={":one": 1},
            )
        return result

    with patch.object(service, "_users_table", users):
        with patch.object(users, "get_item", side_effect=racing_get_item):
            balance = service.grant_with_claim(REF_KEY, USER_ID, 10, reason="purchase")

    assert calls["n"] == 2
    assert balance == 12
    assert int(users.get_item(Key={"user_id": USER_ID})["Item"]["credits"]) == 12
    rows = _ledger_rows(ddb)
    assert len(rows) == 1
    assert int(rows[0]["balance_after"]) == 12


def test_transaction_failure_leaves_no_marker(ddb):
    """트랜잭션이 통째로 실패하면 마커도 남지 않아 같은 토큰으로 재시도 가능"""
    service = CreditService()
    client = service.users_table.meta.client
    original = client.transact_write_items
    attempts = {"n": 0}

    def flaky(**kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("Lambda 중단 흉내")
        return original(**kwargs)

    with patch.object(client, "transact_write_items", side_effect=flaky):
        with pytest.raises(RuntimeError):
            service.grant_with_claim(REF_KEY, USER_ID, 10, reason="purchase")
        assert _claim(ddb) is None

        balance = service.grant_with_claim(REF_KEY, USER_ID, 10, reason="purchase")

    assert balance == 13


def test_remove_attrs_cleared_in_same_transaction(ddb):
    """가입 보너스: 지급과 함께 대기 표시가 지워지고, 두 번째 시도는 중복으로 막힌다"""
    users = ddb.Table("hairme-users")
    users.put_item(Item={"user_id": "new", "credits": 0, "signup_bonus_pending": 5})
    service = CreditService()

    balance = service.grant_with_claim(
        "signup_bonus#new",
        "new",
        5,
        reason="signup_bonus",
        remove_attrs=("signup_bonus_pending",),
    )

    assert balance == 5
    item = users.get_item(Key={"user_id": "new"})["Item"]
    assert "signup_bonus_pending" not in item
    with pytest.raises(RefAlreadyClaimedError):
        service.grant_with_claim(
            "signup_bonus#new",
            "new",
            5,
            reason="signup_bonus",
            remove_attrs=("signup_bonus_pending",),
        )
    assert int(users.get_item(Key={"user_id": "new"})["Item"]["credits"]) == 5
