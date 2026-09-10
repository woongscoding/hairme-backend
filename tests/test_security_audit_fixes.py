# -*- coding: utf-8 -*-
"""2026-07 결제·크레딧 보안 감사 수정사항 테스트

- M1: 합성 예외 경로에서 크레딧 환불 보장
- M3: 카카오 토큰 발급 앱(app_id) 검증
- L1: 원장 기록 재시도 + 최종 실패 시 CRITICAL 대사 로그
- L2: ref_key 로그 마스킹
- L4: usage 일일 카운터 복구 (decrement_daily_counter)
"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

import asyncio
import io
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from config.settings import settings
from core.jwt_auth import create_access_token
from main import app
from services.credit_service import CreditService, _mask_ref
from services.kakao_auth_service import KakaoAuthService
from database.user_repository import (
    UserAlreadyExistsError,
    UserRepository,
    kakao_marker_id,
)
from services.usage_limit_service import UsageLimitService


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


# ========== M1: 합성 예외 시 크레딧 환불 ==========


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def synthesis_mocks():
    """크레딧 소비 성공 + 스토리지 비활성(캐시 미스) 상태로 모킹"""
    credit = MagicMock()
    credit.consume.return_value = 4

    storage = MagicMock()
    storage.enabled = False
    storage.build_cache_key.return_value = "cache-key"
    storage.get_cached_result.return_value = None

    synthesis = MagicMock()

    with patch(
        "api.endpoints.synthesis.get_credit_service", return_value=credit
    ), patch(
        "api.endpoints.synthesis.get_photo_storage_service", return_value=storage
    ), patch(
        "api.endpoints.synthesis.get_synthesis_service", return_value=synthesis
    ):
        yield {"credit": credit, "synthesis": synthesis}


class TestSynthesisRefundOnException:
    AUTH = {"Authorization": f"Bearer {create_access_token('synth-user')}"}

    def _post(self, client):
        return client.post(
            "/api/v2/synthesize",
            files={"file": ("face.png", _png_bytes(), "image/png")},
            data={"hairstyle_name": "크롭컷", "gender": "male"},
            headers=self.AUTH,
        )

    def test_exception_after_charge_refunds_credit(self, client, synthesis_mocks):
        """M1: Gemini 예외(타임아웃 등) 시 차감된 크레딧이 환불되어야 함"""
        synthesis_mocks["synthesis"].synthesize_hairstyle.side_effect = Exception(
            "Gemini timeout"
        )

        response = self._post(client)

        assert response.status_code == 500
        synthesis_mocks["credit"].consume.assert_called_once()
        # 환불 = grant(user, cost, reason="refund")
        refund_call = synthesis_mocks["credit"].grant.call_args
        assert refund_call is not None, "예외 경로에서 환불이 호출되지 않음"
        assert refund_call.kwargs.get("reason") == "refund"
        assert "Gemini" not in response.text

    def test_success_does_not_refund(self, client, synthesis_mocks):
        synthesis_mocks["synthesis"].synthesize_hairstyle.return_value = {
            "success": True,
            "image_base64": "aW1n",
            "image_format": "png",
            "message": "done",
        }

        response = self._post(client)

        assert response.status_code == 200
        synthesis_mocks["credit"].grant.assert_not_called()

    def test_soft_failure_still_refunds(self, client, synthesis_mocks):
        """기존 동작 회귀 확인: success=False(422)도 환불"""
        synthesis_mocks["synthesis"].synthesize_hairstyle.return_value = {
            "success": False,
            "message": "얼굴을 찾지 못했습니다",
        }

        response = self._post(client)

        assert response.status_code == 422
        assert (
            synthesis_mocks["credit"].grant.call_args.kwargs.get("reason") == "refund"
        )


# ========== M3: 카카오 app_id 검증 ==========


def _kakao_client(status_code: int, body: dict):
    """httpx.AsyncClient.get 흉내 (async)"""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body

    client = MagicMock()

    async def _get(*args, **kwargs):
        return response

    client.get = MagicMock(side_effect=_get)
    return client


class TestKakaoAppIdVerification:
    def test_token_from_other_app_rejected(self, monkeypatch):
        """M3: 다른 카카오 앱에서 발급된 토큰은 401 (토큰 치환 차단)"""
        monkeypatch.setattr(settings, "KAKAO_APP_ID", "123456")
        service = KakaoAuthService()
        client = _kakao_client(200, {"id": 42, "app_id": 999999})

        with pytest.raises(HTTPException) as exc:
            asyncio.run(service._verify_token_app_id(client, "kakao-token"))
        assert exc.value.status_code == 401

    def test_token_from_our_app_passes(self, monkeypatch):
        monkeypatch.setattr(settings, "KAKAO_APP_ID", "123456")
        service = KakaoAuthService()
        client = _kakao_client(200, {"id": 42, "app_id": 123456})

        asyncio.run(service._verify_token_app_id(client, "kakao-token"))  # no raise

    def test_invalid_token_rejected(self, monkeypatch):
        monkeypatch.setattr(settings, "KAKAO_APP_ID", "123456")
        service = KakaoAuthService()
        client = _kakao_client(401, {"msg": "invalid"})

        with pytest.raises(HTTPException) as exc:
            asyncio.run(service._verify_token_app_id(client, "bad-token"))
        assert exc.value.status_code == 401

    def test_unset_app_id_skips_check(self, monkeypatch):
        """KAKAO_APP_ID 미설정 시 검증 생략 (기존 배포 호환)"""
        monkeypatch.setattr(settings, "KAKAO_APP_ID", "")
        service = KakaoAuthService()
        client = _kakao_client(200, {"id": 42, "app_id": 999999})

        asyncio.run(service._verify_token_app_id(client, "kakao-token"))  # no raise
        client.get.assert_not_called()


# ========== L1: 원장 기록 재시도 / L2: 마스킹 ==========


class TestLedgerReliability:
    def _service_with_tables(self):
        svc = CreditService()
        svc._users_table = MagicMock()
        svc._ledger_table = MagicMock()
        return svc

    def test_ledger_write_retries_once(self):
        svc = self._service_with_tables()
        svc._ledger_table.put_item.side_effect = [Exception("throttled"), None]

        svc._write_ledger("user-1", 5, "purchase", 10, ref_id="token")

        assert svc._ledger_table.put_item.call_count == 2

    def test_ledger_final_failure_logs_critical_without_raising(self, caplog):
        svc = self._service_with_tables()
        svc._ledger_table.put_item.side_effect = Exception("down")

        with caplog.at_level("CRITICAL"):
            svc._write_ledger("user-1", 5, "purchase", 10, ref_id="secret-token")

        record = [r for r in caplog.records if "LEDGER_WRITE_FAILED" in r.message]
        assert record, "최종 실패 시 CRITICAL 대사 로그가 남아야 함"
        # 수동 대사에 필요한 정보 포함 + 토큰 원문 미노출
        assert "user-1" in record[0].message
        assert "secret-token" not in record[0].message

    def test_mask_ref_hides_token_keeps_namespace(self):
        masked = _mask_ref("purchase#very-secret-token")
        assert masked.startswith("purchase#")
        assert "very-secret-token" not in masked
        # 같은 입력은 같은 해시 (로그 상관관계 추적 가능)
        assert masked == _mask_ref("purchase#very-secret-token")


# ========== L4: 일일 카운터 복구 ==========


class TestDecrementDailyCounter:
    def _service_with_table(self):
        svc = UsageLimitService.__new__(UsageLimitService)
        svc._table = MagicMock()
        # table 프로퍼티 접근 우회를 위해 직접 속성 확인
        return svc

    def test_decrement_calls_conditional_update(self):
        svc = UsageLimitService.__new__(UsageLimitService)
        table = MagicMock()
        with patch.object(UsageLimitService, "table", new=table):
            svc.decrement_daily_counter("reward_ad#user-1")

        kwargs = table.update_item.call_args.kwargs
        assert kwargs["Key"]["device_id"] == "reward_ad#user-1"
        # 0 밑으로 내려가지 않도록 조건부
        assert "#cnt > :zero" in kwargs["ConditionExpression"]

    def test_decrement_swallows_errors(self):
        """복구 실패는 전파하지 않음 (best effort)"""
        svc = UsageLimitService.__new__(UsageLimitService)
        table = MagicMock()
        table.update_item.side_effect = ClientError(
            {"Error": {"Code": "ConditionalCheckFailedException", "Message": "x"}},
            "UpdateItem",
        )
        with patch.object(UsageLimitService, "table", new=table):
            svc.decrement_daily_counter("reward_ad#user-1")  # no raise


# ========== H3: 카카오 가입 경합 (kakao_id 유일성 트랜잭션) ==========
class TestUserCreateUniqueness:
    def _repo_with_table(self):
        repo = UserRepository()
        table = MagicMock()
        table.name = "hairme-users"
        repo._table = table
        return repo, table

    def test_create_writes_marker_and_user_in_one_transaction(self):
        repo, table = self._repo_with_table()

        user = repo.create(kakao_id="12345678", nickname="테스트유저")

        items = table.meta.client.transact_write_items.call_args.kwargs["TransactItems"]
        assert len(items) == 2
        marker_put, user_put = items[0]["Put"], items[1]["Put"]
        # 리소스 클라이언트(table.meta.client)는 파이썬 값을 자동 직렬화하므로
        # Item 은 평문 값이어야 한다. {"S": ...} 로 미리 감싸면 Map(M)으로 이중
        # 직렬화되어 "Type mismatch for key user_id expected: S actual: M" 가 난다
        # (2026-09-09 프로덕션 신규 가입 전면 실패 회귀).
        assert marker_put["Item"]["user_id"] == kakao_marker_id("12345678")
        assert isinstance(marker_put["Item"]["user_id"], str)
        assert marker_put["Item"]["ref_user_id"] == user["user_id"]
        assert marker_put["ConditionExpression"] == "attribute_not_exists(user_id)"
        assert user_put["Item"]["user_id"] == user["user_id"]
        assert isinstance(user_put["Item"]["user_id"], str)
        assert user_put["Item"]["kakao_id"] == "12345678"
        assert user_put["ConditionExpression"] == "attribute_not_exists(user_id)"
        # 마커에는 kakao_id 속성이 없어야 GSI에 색인되지 않는다
        assert "kakao_id" not in marker_put["Item"]

    def test_create_raises_user_already_exists_on_marker_conflict(self):
        repo, table = self._repo_with_table()
        table.meta.client.transact_write_items.side_effect = ClientError(
            {
                "Error": {
                    "Code": "TransactionCanceledException",
                    "Message": "Transaction cancelled",
                },
                "CancellationReasons": [
                    {"Code": "ConditionalCheckFailed"},
                    {"Code": "None"},
                ],
            },
            "TransactWriteItems",
        )

        with pytest.raises(UserAlreadyExistsError):
            repo.create(kakao_id="12345678", nickname="테스트유저")

    def test_create_reraises_other_client_errors(self):
        repo, table = self._repo_with_table()
        table.meta.client.transact_write_items.side_effect = ClientError(
            {
                "Error": {
                    "Code": "ProvisionedThroughputExceededException",
                    "Message": "x",
                }
            },
            "TransactWriteItems",
        )

        with pytest.raises(ClientError):
            repo.create(kakao_id="12345678", nickname="테스트유저")

    def test_get_by_kakao_marker_uses_consistent_read(self):
        repo, table = self._repo_with_table()
        table.get_item.side_effect = [
            {"Item": {"user_id": kakao_marker_id("12345678"), "ref_user_id": "u-1"}},
            {"Item": {"user_id": "u-1", "credits": 3}},
        ]

        user = repo.get_by_kakao_marker("12345678")

        assert user["user_id"] == "u-1"
        first_kwargs = table.get_item.call_args_list[0].kwargs
        assert first_kwargs["ConsistentRead"] is True
        assert first_kwargs["Key"]["user_id"] == "kakao#12345678"

    def test_get_by_kakao_marker_returns_none_when_absent(self):
        repo, table = self._repo_with_table()
        table.get_item.return_value = {}

        assert repo.get_by_kakao_marker("12345678") is None
