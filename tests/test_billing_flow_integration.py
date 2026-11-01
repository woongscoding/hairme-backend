"""Play 결제 → 크레딧 → 합성 통합 테스트 (moto 가짜 DynamoDB) - 수익화 v2.1 Phase 0

단위 테스트는 CreditService 를 MagicMock 으로 갈아끼워서 DynamoDB 조건식이 실제로
의도대로 동작하는지는 확인하지 못한다. 여기서는 moto 위에 실제 테이블을 만들고
엔드포인트 → CreditService → DynamoDB 까지 그대로 태운다. Play API 와 Gemini 만 가짜다.

흐름: 구매 검증 → 크레딧 지급(유료분) → 합성 차감(유료 1K) → 실패 시 자동 환불
      → Play 환불 시 회수(void reclaim) → 유료분 소진 후 무료 512px+워터마크
"""

import base64
import io
import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

from decimal import Decimal
from unittest.mock import MagicMock, patch

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
from PIL import Image

import services.credit_service as credit_module
import services.usage_limit_service as usage_module
from api.endpoints.credits import expected_obfuscated_account_id
from core.jwt_auth import create_access_token
from main import app
from services.credit_service import CreditService
from services.play_void_reclaim_service import PlayVoidReclaimService
from services.usage_limit_service import UsageLimitService

pytestmark = pytest.mark.moto

REGION = "ap-northeast-2"
USER_ID = "buyer-0001"


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (180, 140, 120)).save(buf, format="JPEG")
    return buf.getvalue()


def _png_b64(size=(96, 96)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 120, 200)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


GEMINI_RESULT_B64 = _png_b64()


def _create_tables(ddb) -> None:
    ddb.create_table(
        TableName="hairme-users",
        KeySchema=[{"AttributeName": "user_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "user_id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    ddb.create_table(
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
    ddb.create_table(
        TableName="hairstyle_usage",
        KeySchema=[
            {"AttributeName": "device_id", "KeyType": "HASH"},
            {"AttributeName": "date", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "device_id", "AttributeType": "S"},
            {"AttributeName": "date", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )


class FakePlay:
    """Play Developer API 대역: 모든 토큰을 유효한 미승인 구매로 본다"""

    package_name = "com.hairme.test"
    session = object()

    def __init__(self):
        self.acknowledged = []
        self.voided = []
        # token -> obfuscatedExternalAccountId (없으면 계정 연결 이전 결제)
        self.account_ids = {}

    def verify_product_purchase(self, product_id, token):
        return {
            "order_id": f"GPA.{token}",
            "acknowledgement_state": 0,
            "obfuscated_account_id": self.account_ids.get(token),
        }

    def acknowledge_product_purchase(self, product_id, token):
        self.acknowledged.append(token)

    def list_voided_purchases(self, start_time_millis, page_token=None):
        return {"voidedPurchases": [{"purchaseToken": t} for t in self.voided]}


@pytest.fixture(autouse=True)
def no_rate_limit(monkeypatch):
    """분당 호출 제한은 테스트 간에 공유되므로 끈다 (기존 합성 테스트와 같은 방식)"""
    from api.endpoints import credits, synthesis

    for module in (synthesis, credits):
        monkeypatch.setattr(module.limiter, "enabled", False)


@pytest.fixture
def env():
    with mock_aws():
        ddb = boto3.resource("dynamodb", region_name=REGION)
        _create_tables(ddb)
        ddb.Table("hairme-users").put_item(
            Item={"user_id": USER_ID, "credits": 5}  # 가입 보너스 5 (무료분)
        )

        credit_service = CreditService()
        usage_service = UsageLimitService()
        play = FakePlay()
        synthesis = MagicMock()
        synthesis.synthesize_hairstyle.return_value = {
            "success": True,
            "image_base64": GEMINI_RESULT_B64,
            "image_format": "png",
            "message": "ok",
            "api_calls": 1,
        }
        storage = MagicMock()
        storage.enabled = False
        storage.get_cached_result.return_value = None
        storage.build_cache_key.side_effect = lambda *parts: "k" * 64

        with patch.object(
            credit_module, "_credit_service", credit_service
        ), patch.object(usage_module, "_usage_limit_service", usage_service), patch(
            "api.endpoints.credits.get_play_billing_service", return_value=play
        ), patch(
            "api.endpoints.synthesis.get_synthesis_service", return_value=synthesis
        ), patch(
            "api.endpoints.synthesis.get_photo_storage_service", return_value=storage
        ), patch(
            "api.endpoints.synthesis._safe_product_recommendations", return_value=[]
        ), TestClient(
            app
        ) as client:
            yield {
                "client": client,
                "ddb": ddb,
                "credit": credit_service,
                "play": play,
                "synthesis": synthesis,
                "headers": {"Authorization": f"Bearer {create_access_token(USER_ID)}"},
            }


def _user(env) -> dict:
    return env["ddb"].Table("hairme-users").get_item(Key={"user_id": USER_ID})["Item"]


def _ledger_reasons(env) -> list:
    items = (
        env["ddb"]
        .Table("hairme-credit-ledger")
        .query(
            KeyConditionExpression="user_id = :u",
            ExpressionAttributeValues={":u": USER_ID},
        )["Items"]
    )
    return [item["reason"] for item in items]


def _purchase(env, product_id, token):
    return env["client"].post(
        "/api/credits/purchase",
        json={"product_id": product_id, "purchase_token": token},
        headers=env["headers"],
    )


def _synthesize(env):
    return env["client"].post(
        "/api/v2/synthesize",
        files={"file": ("face.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"hairstyle_name": "투블럭컷", "gender": "male"},
        headers=env["headers"],
    )


class TestBillingFlow:
    def test_full_flow_purchase_consume_refund_reclaim(self, env):
        # 1. 구매 검증 → 승인 → 유료분 지급
        response = _purchase(env, "credits_10", "tok-10")
        assert response.status_code == 200, response.text
        assert response.json()["balance"] == 15
        assert env["play"].acknowledged == ["tok-10"]
        user = _user(env)
        assert user["credits"] == Decimal(15)
        assert user["paid_credits"] == Decimal(10)
        assert user.get("first_purchase_at")

        # 같은 토큰 재전송은 중복 지급하지 않는다
        assert _purchase(env, "credits_10", "tok-10").status_code == 409
        assert _user(env)["credits"] == Decimal(15)

        # 2. 합성 차감 - 유료분부터, 1K·워터마크 없음
        response = _synthesize(env)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["quota"] == {"mode": "credits", "balance": 14, "tier": "paid"}
        assert body["quality"] == {
            "tier": "paid",
            "image_size": "1K",
            "watermarked": False,
        }
        assert body["is_preview"] is False
        assert body["image_base64"] == GEMINI_RESULT_B64
        assert (
            env["synthesis"].synthesize_hairstyle.call_args.kwargs["image_size"] == "1K"
        )
        assert _user(env)["paid_credits"] == Decimal(9)

        # 3. 합성 실패(이미지 미생성) → 같은 유료분으로 자동 환불
        env["synthesis"].synthesize_hairstyle.return_value = {
            "success": False,
            "image_base64": None,
            "image_format": None,
            "message": "AI가 잠깐 졸았나봐요..",
            "api_calls": 3,
        }
        response = _synthesize(env)
        assert response.status_code == 422
        assert response.json()["refunded"] is True
        assert response.json()["balance"] == 14
        user = _user(env)
        assert user["credits"] == Decimal(14)
        assert user["paid_credits"] == Decimal(9)

        # 4. 합성 중 예외(타임아웃 등) → 500 + 자동 환불
        env["synthesis"].synthesize_hairstyle.side_effect = TimeoutError("gemini")
        assert _synthesize(env).status_code == 500
        assert _user(env)["credits"] == Decimal(14)
        assert _user(env)["paid_credits"] == Decimal(9)

        # 5. Play 환불 → 회수 (구매 10 중 1 사용: 잔액 14-10=4, 유료분 0)
        env["play"].voided = ["tok-10"]
        reclaimer = PlayVoidReclaimService(
            play_service=env["play"], credit_service=env["credit"]
        )
        summary = reclaimer.run()
        assert summary["reclaimed"] == 1
        user = _user(env)
        assert user["credits"] == Decimal(4)
        assert user["paid_credits"] == Decimal(0)

        # 다음 폴링에서 같은 환불을 다시 회수하지 않는다
        assert reclaimer.run()["already_reclaimed"] == 1
        assert _user(env)["credits"] == Decimal(4)

        # 6. 유료분이 없으면 무료 등급 (512px + 워터마크)
        env["synthesis"].synthesize_hairstyle.side_effect = None
        env["synthesis"].synthesize_hairstyle.return_value = {
            "success": True,
            "image_base64": GEMINI_RESULT_B64,
            "image_format": "png",
            "message": "ok",
            "api_calls": 1,
        }
        response = _synthesize(env)
        assert response.status_code == 200
        body = response.json()
        assert body["quota"]["tier"] == "free"
        assert body["quality"]["watermarked"] is True
        assert body["is_preview"] is True
        assert body["image_base64"] != GEMINI_RESULT_B64  # 워터마크가 들어감
        assert (
            env["synthesis"].synthesize_hairstyle.call_args.kwargs["image_size"]
            == "512"
        )
        assert _user(env)["credits"] == Decimal(3)

        reasons = _ledger_reasons(env)
        assert reasons.count("purchase") == 1
        assert reasons.count("synthesis_paid") == 3  # 성공 1 + 실패 2 (환불됨)
        assert reasons.count("refund") == 2
        assert reasons.count("void_reclaim") == 1
        assert reasons.count("synthesis") == 1

    def test_reclaim_after_all_used_goes_negative_and_blocks(self, env):
        assert _purchase(env, "credits_10", "tok-a").status_code == 200
        env["ddb"].Table("hairme-users").update_item(
            Key={"user_id": USER_ID},
            UpdateExpression="SET credits = :c, paid_credits = :p",
            ExpressionAttributeValues={":c": 0, ":p": 0},
        )
        env["play"].voided = ["tok-a"]
        PlayVoidReclaimService(
            play_service=env["play"], credit_service=env["credit"]
        ).run()
        assert _user(env)["credits"] == Decimal(-10)

        response = _synthesize(env)
        assert response.status_code == 402
        env["synthesis"].synthesize_hairstyle.assert_not_called()

    def test_starter_once_per_account(self, env):
        response = _purchase(env, "starter_5", "tok-s1")
        assert response.status_code == 200
        assert response.json()["granted"] == 5

        # 같은 토큰 재시도는 계정당 1회 검사를 통과하고 중복 지급에서 막힌다
        assert _purchase(env, "starter_5", "tok-s1").status_code == 409

        # 다른 토큰으로 재구매: 승인하지 않고 거절 (Google 이 자동 환불)
        response = _purchase(env, "starter_5", "tok-s2")
        assert response.status_code == 422
        assert response.json()["error"] == "one_per_account_used"
        assert "tok-s2" not in env["play"].acknowledged
        assert _user(env)["credits"] == Decimal(10)

        products = (
            env["client"].get("/api/credits/products", headers=env["headers"]).json()
        )
        starter = next(
            p for p in products["products"] if p["product_id"] == "starter_5"
        )
        assert starter["available"] is False
        assert products["ad_free"] is True
        assert products["paid_balance"] == 5

    def test_purchase_linked_to_other_account_is_rejected(self, env):
        """결제 때 넣은 계정 값이 다르면 403 - 첫 구매 마커·승인·지급 전부 없음"""
        env["play"].account_ids["tok-other"] = expected_obfuscated_account_id(
            "someone-else"
        )
        response = _purchase(env, "starter_5", "tok-other")
        assert response.status_code == 403
        assert response.json()["error"] == "purchase_account_mismatch"
        assert "tok-other" not in env["play"].acknowledged  # Google 자동 환불
        assert _user(env)["credits"] == Decimal(5)
        assert "paid_credits" not in _user(env)

        # 마커를 선점하지 않았으므로 본인 계정의 첫 구매는 정상 처리
        env["play"].account_ids["tok-mine"] = expected_obfuscated_account_id(USER_ID)
        response = _purchase(env, "starter_5", "tok-mine")
        assert response.status_code == 200, response.text
        assert response.json()["granted"] == 5
        assert "tok-mine" in env["play"].acknowledged

    def test_purchase_without_account_id_still_passes(self, env):
        """계정 연결 이전 앱의 결제(값 없음)는 그대로 통과"""
        assert "tok-legacy" not in env["play"].account_ids
        assert _purchase(env, "credits_10", "tok-legacy").status_code == 200

    def test_retired_credits_100_rejected(self, env):
        assert _purchase(env, "credits_100", "tok-x").status_code == 400
