"""미용실용 앱(Hair Consultant) 출시 전 수정 테스트

1. 회원 탈퇴 API (Pro 계정만): 결과 사진·공유 연결 정보 삭제 → 회원 행·카카오 마커 삭제
2. QR 용 짧은 공유 주소 (/s/{token}): 7일 동안만, 추측할 수 없는 토큰
3. 합성 분당 제한: Pro 계정은 계정별, 손님·비로그인은 기존 IP 별
4. 체형 코디: Pro 계정 거절
손님 계정 요청의 동작은 기존과 같은지 함께 확인한다.
"""

import json
import os
from datetime import datetime, timedelta, timezone

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from starlette.requests import Request

from config.settings import settings
from core.jwt_auth import create_access_token
from core.account_type import ACCOUNT_PRO
from core.rate_limit import pro_account_or_ip_key
from services.photo_storage_service import (
    PhotoStorageService,
    ShareLinkExpired,
    share_pointer_key,
    share_token_of_key,
)
from tests import test_pro_account as base
from tests.test_pro_account import (
    CUSTOMER_USER,
    PRO_USER,
    _customer_auth,
    _png_bytes,
    _pro_auth,
    _synthesize,
)

# test_pro_account 의 픽스처를 그대로 쓴다
client = base.client
synthesis_env = base.synthesis_env

TOKEN = "AbCdEfGhIjKlMnOpQrSt_-"  # 22자


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "Op")


def _storage(monkeypatch) -> PhotoStorageService:
    monkeypatch.setenv("PHOTO_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(settings, "PHOTO_S3_BUCKET", "test-bucket")
    svc = PhotoStorageService()
    svc._client = MagicMock()
    svc._client.generate_presigned_url.return_value = "https://s3/presigned?a=1&b=2"
    return svc


# ========== 1. 회원 탈퇴 ==========


@pytest.fixture
def withdraw_env():
    from api.endpoints import auth

    previous = auth.limiter.enabled
    auth.limiter.enabled = False
    repo = MagicMock()
    repo.get_by_id_consistent.return_value = dict(PRO_USER)
    storage = MagicMock()
    storage.delete_user_results.return_value = 3
    with patch("api.endpoints.auth.get_user_repository", return_value=repo), patch(
        "api.endpoints.auth.get_photo_storage_service", return_value=storage
    ), patch("api.endpoints.auth.log_structured") as log:
        yield {"repo": repo, "storage": storage, "log": log}
    auth.limiter.enabled = previous


class TestWithdraw:
    def test_pro_withdraw_deletes_results_then_account(self, client, withdraw_env):
        env = withdraw_env
        response = client.delete("/api/auth/me", headers=_pro_auth())
        assert response.status_code == 200, response.text
        assert response.json() == {"success": True, "deleted_results": 3}
        env["storage"].delete_user_results.assert_called_once_with("pro-user-id", True)
        env["repo"].delete_account.assert_called_once_with(
            "pro-user-id", PRO_USER["kakao_id"]
        )
        event = env["log"].call_args.args
        assert event[0] == "account_withdrawn"
        assert event[1]["deleted_results"] == 3
        assert event[1]["user_id"] != "pro-user-id"  # 마스킹

    def test_photo_delete_failure_keeps_account(self, client, withdraw_env):
        env = withdraw_env
        env["storage"].delete_user_results.side_effect = RuntimeError("s3 down")
        response = client.delete("/api/auth/me", headers=_pro_auth())
        assert response.status_code == 503
        assert response.json()["error"] == "withdrawal_failed"
        env["repo"].delete_account.assert_not_called()

    def test_account_delete_failure_is_503(self, client, withdraw_env):
        withdraw_env["repo"].delete_account.side_effect = RuntimeError("ddb down")
        response = client.delete("/api/auth/me", headers=_pro_auth())
        assert response.status_code == 503

    def test_already_withdrawn_is_success(self, client, withdraw_env):
        env = withdraw_env
        env["repo"].get_by_id_consistent.return_value = None
        env["storage"].delete_user_results.return_value = 0
        response = client.delete("/api/auth/me", headers=_pro_auth())
        assert response.status_code == 200
        env["storage"].delete_user_results.assert_called_once()
        env["repo"].delete_account.assert_not_called()

    def test_concurrent_delete_is_success(self, client, withdraw_env):
        withdraw_env["repo"].delete_account.side_effect = ValueError("gone")
        assert client.delete("/api/auth/me", headers=_pro_auth()).status_code == 200

    def test_customer_not_supported(self, client, withdraw_env):
        env = withdraw_env
        response = client.delete("/api/auth/me", headers=_customer_auth())
        assert response.status_code == 403
        assert response.json()["error"] == "withdrawal_not_supported"
        env["storage"].delete_user_results.assert_not_called()
        env["repo"].delete_account.assert_not_called()

    def test_pro_token_but_customer_row_not_supported(self, client, withdraw_env):
        withdraw_env["repo"].get_by_id_consistent.return_value = dict(CUSTOMER_USER)
        response = client.delete("/api/auth/me", headers=_pro_auth())
        assert response.status_code == 403
        withdraw_env["storage"].delete_user_results.assert_not_called()

    def test_requires_login(self, client, withdraw_env):
        assert client.delete("/api/auth/me").status_code == 401


class TestDeleteAccountRepository:
    def _repo(self):
        from database.user_repository import UserRepository

        repo = UserRepository()
        table = MagicMock()
        table.name = "hairme-users"
        repo._table = table
        return repo, table

    def test_deletes_user_and_own_marker_together(self):
        from database.user_repository import UserRepository

        repo, table = self._repo()
        with patch.object(UserRepository, "table", table):
            repo.delete_account("u1", "900001")
        items = table.meta.client.transact_write_items.call_args.kwargs["TransactItems"]
        assert items[0]["Delete"]["Key"] == {"user_id": "u1"}
        assert items[0]["Delete"]["ConditionExpression"] == "attribute_exists(user_id)"
        assert items[1]["Delete"]["Key"] == {"user_id": "kakao#900001"}
        # 다른 사람의 마커는 지우지 않는다
        assert "ref_user_id = :uid" in items[1]["Delete"]["ConditionExpression"]
        assert items[1]["Delete"]["ExpressionAttributeValues"] == {":uid": "u1"}

    def test_missing_user_is_value_error(self):
        from database.user_repository import UserRepository

        repo, table = self._repo()
        error = ClientError(
            {
                "Error": {"Code": "TransactionCanceledException", "Message": "x"},
                "CancellationReasons": [
                    {"Code": "ConditionalCheckFailed"},
                    {"Code": "None"},
                ],
            },
            "TransactWriteItems",
        )
        table.meta.client.transact_write_items.side_effect = error
        with patch.object(UserRepository, "table", table):
            with pytest.raises(ValueError):
                repo.delete_account("u1", "900001")


class TestDeleteUserResults:
    def test_deletes_results_and_share_pointers(self, monkeypatch):
        svc = _storage(monkeypatch)
        result_key = f"pro/results/u1/20261110T030000_{TOKEN}.png"
        svc._client.list_objects_v2.side_effect = [
            {
                "Contents": [{"Key": result_key}],
                "IsTruncated": True,
                "NextContinuationToken": "n",
            },
            {
                "Contents": [{"Key": "pro/results/u1/20261110T030001_old.png"}],
                "IsTruncated": False,
            },
        ]
        svc._client.delete_objects.return_value = {}
        assert svc.delete_user_results("u1", True) == 2

        assert svc._client.list_objects_v2.call_args_list[0].kwargs["Prefix"] == (
            "pro/results/u1/"
        )
        deleted = [
            o["Key"]
            for o in svc._client.delete_objects.call_args.kwargs["Delete"]["Objects"]
        ]
        assert deleted == [
            result_key,
            "pro/results/u1/20261110T030001_old.png",
            share_pointer_key(TOKEN),
        ]

    def test_partial_failure_raises(self, monkeypatch):
        svc = _storage(monkeypatch)
        svc._client.list_objects_v2.return_value = {
            "Contents": [{"Key": "pro/results/u1/a_b.png"}],
            "IsTruncated": False,
        }
        svc._client.delete_objects.return_value = {"Errors": [{"Key": "x"}]}
        with pytest.raises(RuntimeError):
            svc.delete_user_results("u1", True)

    def test_nothing_to_delete(self, monkeypatch):
        svc = _storage(monkeypatch)
        svc._client.list_objects_v2.return_value = {"IsTruncated": False}
        assert svc.delete_user_results("u1", True) == 0
        svc._client.delete_objects.assert_not_called()


# ========== 2. 짧은 공유 주소 ==========


class TestSaveProResult:
    def test_saves_result_and_pointer(self, monkeypatch):
        svc = _storage(monkeypatch)
        saved = svc.save_pro_result("u1", "aW1n", "png", "레이어드 컷")
        token = saved["share_token"]
        assert len(token) == 22
        result_call, pointer_call = svc._client.put_object.call_args_list
        result_key = result_call.kwargs["Key"]
        assert result_key.startswith("pro/results/u1/")
        assert share_token_of_key(result_key) == token
        assert pointer_call.kwargs["Key"] == f"pro/results/_share/{token}.json"
        pointer = json.loads(pointer_call.kwargs["Body"])
        assert pointer["key"] == result_key
        assert datetime.fromisoformat(pointer["created_at"]) == saved["created_at"]
        assert saved["result_url"] == "https://s3/presigned?a=1&b=2"

    def test_tokens_are_random(self, monkeypatch):
        svc = _storage(monkeypatch)
        tokens = {svc.save_pro_result("u1", "aW1n")["share_token"] for _ in range(50)}
        assert len(tokens) == 50

    def test_pointer_failure_keeps_result(self, monkeypatch):
        svc = _storage(monkeypatch)
        svc._client.put_object.side_effect = [None, RuntimeError("s3")]
        saved = svc.save_pro_result("u1", "aW1n")
        assert saved["share_token"] is None
        assert saved["result_url"]

    def test_result_failure_is_none(self, monkeypatch):
        svc = _storage(monkeypatch)
        svc._client.put_object.side_effect = RuntimeError("s3")
        assert svc.save_pro_result("u1", "aW1n") is None


class TestResolveShareToken:
    def _pointer(self, svc, created_at, key=None):
        body = MagicMock()
        body.read.return_value = json.dumps(
            {
                "key": key or f"pro/results/u1/20261110T030000_{TOKEN}.png",
                "created_at": created_at.isoformat(),
            }
        ).encode()
        svc._client.get_object.return_value = {"Body": body}

    def test_open_within_retention(self, monkeypatch):
        svc = _storage(monkeypatch)
        created = datetime.now(timezone.utc) - timedelta(days=6, hours=23)
        self._pointer(svc, created)
        found = svc.resolve_share_token(TOKEN)
        assert found["image_url"] == "https://s3/presigned?a=1&b=2"
        assert found["expires_at"] == created + timedelta(days=7)
        assert svc._client.get_object.call_args.kwargs["Key"] == (
            f"pro/results/_share/{TOKEN}.json"
        )
        # 열 때마다 짧은 presigned URL 을 새로 만든다
        assert (
            svc._client.generate_presigned_url.call_args.kwargs["ExpiresIn"]
            == settings.PRO_SHARE_IMAGE_URL_SECONDS
        )

    def test_expired_after_retention(self, monkeypatch):
        svc = _storage(monkeypatch)
        self._pointer(svc, datetime.now(timezone.utc) - timedelta(days=7, seconds=1))
        with pytest.raises(ShareLinkExpired):
            svc.resolve_share_token(TOKEN)
        svc._client.generate_presigned_url.assert_not_called()

    @pytest.mark.parametrize(
        "token", ["", "short", TOKEN + "x", "../../results/u1/a.png!!", "가" * 22]
    )
    def test_malformed_token_never_hits_s3(self, monkeypatch, token):
        svc = _storage(monkeypatch)
        assert svc.resolve_share_token(token) is None
        svc._client.get_object.assert_not_called()

    def test_unknown_token(self, monkeypatch):
        svc = _storage(monkeypatch)
        svc._client.get_object.side_effect = _client_error("NoSuchKey")
        assert svc.resolve_share_token(TOKEN) is None

    def test_pointer_outside_pro_results_rejected(self, monkeypatch):
        svc = _storage(monkeypatch)
        self._pointer(svc, datetime.now(timezone.utc), key="results/u1/a.png")
        assert svc.resolve_share_token(TOKEN) is None
        svc._client.generate_presigned_url.assert_not_called()

    def test_deleted_result(self, monkeypatch):
        svc = _storage(monkeypatch)
        self._pointer(svc, datetime.now(timezone.utc))
        svc._client.head_object.side_effect = _client_error("404")
        assert svc.resolve_share_token(TOKEN) is None

    def test_s3_error_propagates(self, monkeypatch):
        svc = _storage(monkeypatch)
        svc._client.get_object.side_effect = _client_error("InternalError")
        with pytest.raises(ClientError):
            svc.resolve_share_token(TOKEN)

    def test_share_token_of_key(self):
        assert share_token_of_key(f"pro/results/u1/20261110T030000_{TOKEN}.png") == (
            TOKEN
        )
        assert share_token_of_key("pro/results/u1/20261110T030000_ab12cd34.png") is None
        assert share_token_of_key("pro/results/u1/noseparator.png") is None


@pytest.fixture
def share_storage():
    from api.endpoints import share

    previous = share.limiter.enabled
    share.limiter.enabled = False
    storage = MagicMock()
    with patch("api.endpoints.share.get_photo_storage_service", return_value=storage):
        yield storage
    share.limiter.enabled = previous


class TestSharePage:
    def test_open_page(self, client, share_storage):
        share_storage.resolve_share_token.return_value = {
            "image_url": "https://s3/presigned?a=1&b=2",
            "expires_at": datetime(2026, 11, 17, 3, 0, tzinfo=timezone.utc),
        }
        response = client.get(f"/s/{TOKEN}")  # 로그인 없이 열린다
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert response.headers["cache-control"] == "no-store"
        assert "noindex" in response.headers["x-robots-tag"]
        # 이미지 요청(다른 출처)에 토큰이 담긴 경로를 실어 보내지 않는다
        assert response.headers["referrer-policy"] == "strict-origin-when-cross-origin"
        html = response.text
        assert 'src="https://s3/presigned?a=1&amp;b=2"' in html
        assert "11월 17일 12:00까지" in html  # 한국 시간
        share_storage.resolve_share_token.assert_called_once_with(TOKEN)

    def test_expired(self, client, share_storage):
        share_storage.resolve_share_token.side_effect = ShareLinkExpired()
        response = client.get(f"/s/{TOKEN}")
        assert response.status_code == 410
        assert "보관 기간" in response.text

    def test_not_found(self, client, share_storage):
        share_storage.resolve_share_token.return_value = None
        assert client.get(f"/s/{TOKEN}").status_code == 404

    def test_storage_error(self, client, share_storage):
        share_storage.resolve_share_token.side_effect = RuntimeError("s3")
        assert client.get(f"/s/{TOKEN}").status_code == 503

    def test_token_not_logged(self, client, share_storage):
        share_storage.resolve_share_token.return_value = None
        with patch("api.endpoints.share.log_structured") as log:
            client.get(f"/s/{TOKEN}")
        assert TOKEN not in str(log.call_args_list)


@pytest.mark.usefixtures("no_synthesis_limit")
class TestSynthesisShareUrl:
    def test_pro_response_has_share_url(self, client, synthesis_env):
        body = _synthesize(client, _pro_auth()).json()
        assert body["share_url"] == "http://testserver/s/" + "T" * 22
        assert body["share_expires_at"] == "2026-11-17T03:00:00+00:00"
        assert body["result_url"] == "https://s3/result.png"

    def test_base_url_override(self, client, synthesis_env, monkeypatch):
        monkeypatch.setattr(settings, "PRO_SHARE_BASE_URL", "https://hc.example/")
        body = _synthesize(client, _pro_auth()).json()
        assert body["share_url"] == "https://hc.example/s/" + "T" * 22

    def test_pointer_failure_gives_null_share_url(self, client, synthesis_env):
        synthesis_env["storage"].save_pro_result.return_value = {
            "result_url": "https://s3/result.png",
            "share_token": None,
            "created_at": datetime.now(timezone.utc),
        }
        body = _synthesize(client, _pro_auth()).json()
        assert body["share_url"] is None
        assert body["share_expires_at"] is None

    def test_customer_response_has_no_share_keys(self, client, synthesis_env):
        body = _synthesize(client, _customer_auth()).json()
        assert "share_url" not in body
        assert "share_expires_at" not in body


# ========== 3. 합성 분당 제한 ==========


@pytest.fixture
def no_synthesis_limit():
    from api.endpoints import synthesis

    previous = synthesis.limiter.enabled
    synthesis.limiter.enabled = False
    yield
    synthesis.limiter.enabled = previous


@pytest.fixture
def fresh_synthesis_limit():
    from api.endpoints import synthesis

    previous = synthesis.limiter.enabled
    synthesis.limiter.enabled = True
    synthesis.limiter.reset()
    yield
    synthesis.limiter.reset()
    synthesis.limiter.enabled = previous


def _request(headers: dict) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": raw,
            "client": ("10.0.0.1", 1234),
        }
    )


class TestRateLimitKey:
    def test_pro_token_is_per_account(self):
        assert pro_account_or_ip_key(_request(_pro_auth("p1"))) == "pro-user:p1"

    def test_customer_token_is_ip(self):
        assert pro_account_or_ip_key(_request(_customer_auth())) == "10.0.0.1"

    def test_no_token_is_ip(self):
        assert pro_account_or_ip_key(_request({})) == "10.0.0.1"

    def test_invalid_token_is_ip(self):
        headers = {"Authorization": "Bearer not-a-jwt"}
        assert pro_account_or_ip_key(_request(headers)) == "10.0.0.1"

    def test_expired_pro_token_is_ip(self):
        with patch.object(settings, "JWT_ACCESS_TOKEN_EXPIRE_MINUTES", -1):
            token = create_access_token("p1", 1, ACCOUNT_PRO)
        headers = {"Authorization": f"Bearer {token}"}
        assert pro_account_or_ip_key(_request(headers)) == "10.0.0.1"


@pytest.mark.usefixtures("fresh_synthesis_limit")
class TestSynthesisRateLimit:
    def test_pro_accounts_on_same_ip_do_not_share_limit(self, client, synthesis_env):
        for _ in range(5):
            assert _synthesize(client, _pro_auth("designer-a")).status_code == 200
        # 같은 IP 의 다른 디자이너는 자기 몫 5회를 그대로 쓴다
        for _ in range(5):
            assert _synthesize(client, _pro_auth("designer-b")).status_code == 200
        # 각 계정의 6번째는 막힌다
        assert _synthesize(client, _pro_auth("designer-a")).status_code == 429
        assert _synthesize(client, _pro_auth("designer-b")).status_code == 429

    def test_pro_requests_do_not_use_customer_ip_limit(self, client, synthesis_env):
        for _ in range(5):
            assert _synthesize(client, _pro_auth("designer-a")).status_code == 200
        for _ in range(5):
            assert _synthesize(client, _customer_auth()).status_code == 200
        response = _synthesize(client, _customer_auth())
        assert response.status_code == 429
        assert response.json()["error"] == "rate_limit_exceeded"

    def test_customers_on_same_ip_still_share_limit(self, client, synthesis_env):
        """손님 계정·비로그인은 기존처럼 IP 기준 (계정이 달라도 같이 센다)"""
        for i in range(5):
            response = _synthesize(client, _customer_auth(f"customer-{i}"))
            assert response.status_code == 200
        assert _synthesize(client, _customer_auth("customer-x")).status_code == 429


# ========== 4. 체형 코디 ==========


class TestFashionRejectsPro:
    def _post(self, client, headers):
        return client.post(
            "/api/fashion/body-analyze",
            files={"file": ("body.png", _png_bytes(), "image/png")},
            data={"gender": "female"},
            headers=headers,
        )

    def test_pro_rejected_before_photo_is_read(self, client):
        from api.endpoints import fashion

        previous = fashion.limiter.enabled
        fashion.limiter.enabled = False
        try:
            with patch(
                "api.endpoints.fashion.get_photo_storage_service"
            ) as storage, patch(
                "api.endpoints.fashion.get_body_pose_analyzer"
            ) as analyzer:
                response = self._post(client, _pro_auth())
        finally:
            fashion.limiter.enabled = previous
        assert response.status_code == 403
        assert response.json() == {
            "error": "pro_not_supported",
            "message": "이 계정에서는 체형 코디를 사용할 수 없습니다.",
        }
        storage.assert_not_called()
        analyzer.assert_not_called()

    def test_customer_still_analyzed(self, client):
        from api.endpoints import fashion

        previous = fashion.limiter.enabled
        fashion.limiter.enabled = False
        try:
            with patch(
                "api.endpoints.fashion.get_body_pose_analyzer",
                side_effect=RuntimeError("stop"),
            ) as analyzer:
                response = self._post(client, _customer_auth())
        finally:
            fashion.limiter.enabled = previous
        # 손님은 거절되지 않고 분석 단계까지 간다 (여기서는 일부러 실패시킴)
        assert response.status_code == 500
        analyzer.assert_called_once()
