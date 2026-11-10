"""미용실용 앱(Hair Consultant) Pro 계정 테스트

S1 계정 구분: 카카오 앱 ID → 계정 유형, 회원 행·JWT 클레임, 토큰 갱신 후 유지, 가입 체험 수량
S2 개인정보: 학습 신호·원본·공용 캐시 차단, pro/results/ 저장, 분석 행 미저장, 염색 거절
S3 결제: 앱별 상품·패키지, 서로의 상품 구매 불가, 환불 회수 잡 두 패키지
S4 그 밖: Pro 워터마크 문구, 제휴 상품 제외, 로그의 계정 유형
그리고 손님 계정 요청의 응답·호출이 기존과 같은지 확인한다.
"""

import asyncio
import io
import os
from datetime import datetime, timezone

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from config.settings import settings
from core.account_type import (
    ACCOUNT_CUSTOMER,
    ACCOUNT_PRO,
    ACCOUNT_TYPE_CLAIM,
    account_type_of_payload,
    account_type_of_user,
)
from core.jwt_auth import (
    TOKEN_TYPE_REFRESH,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_account_type,
)
from main import app
from services.kakao_auth_service import KakaoAuthService

CUSTOMER_APP_ID = "111111"
PRO_APP_ID = "222222"
PRO_PACKAGE = "com.example.pro"
CUSTOMER_PACKAGE = "com.hairme.app"

PRO_USER = {
    "user_id": "pro-user-id",
    "kakao_id": "900001",
    "nickname": "디자이너",
    "credits": 10,
    "training_consent": False,
    "status": "active",
    "token_version": 1,
    "account_type": "pro",
}
CUSTOMER_USER = {
    "user_id": "customer-user-id",
    "kakao_id": "900002",
    "nickname": "손님",
    "credits": 3,
    "training_consent": True,
    "status": "active",
    "token_version": 1,
}


def _pro_auth(user_id: str = "pro-user-id") -> dict:
    return {"Authorization": f"Bearer {create_access_token(user_id, 1, ACCOUNT_PRO)}"}


def _customer_auth(user_id: str = "customer-user-id") -> dict:
    return {"Authorization": f"Bearer {create_access_token(user_id)}"}


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(120, 90, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def no_rate_limits():
    from api.endpoints import auth, credits, events, hair_color, synthesis

    modules = (auth, credits, events, hair_color, synthesis)
    previous = [m.limiter.enabled for m in modules]
    for m in modules:
        m.limiter.enabled = False
    yield
    for m, value in zip(modules, previous):
        m.limiter.enabled = value


# ========== S1. 계정 유형 판정 (카카오 앱 ID) ==========


def _kakao_client(status_code: int, body: dict):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body
    client = MagicMock()

    async def _get(*args, **kwargs):
        return response

    client.get = MagicMock(side_effect=_get)
    return client


class TestKakaoAppAccountType:
    def _verify(self, app_id):
        client = _kakao_client(200, {"id": 42, "app_id": app_id})
        return asyncio.run(KakaoAuthService()._verify_token_app_id(client, "t")), client

    def test_pro_app_token_is_pro(self, monkeypatch):
        monkeypatch.setattr(settings, "KAKAO_APP_ID", CUSTOMER_APP_ID)
        monkeypatch.setattr(settings, "KAKAO_PRO_APP_ID", PRO_APP_ID)
        assert self._verify(int(PRO_APP_ID))[0] == ACCOUNT_PRO

    def test_customer_app_token_is_customer(self, monkeypatch):
        monkeypatch.setattr(settings, "KAKAO_APP_ID", CUSTOMER_APP_ID)
        monkeypatch.setattr(settings, "KAKAO_PRO_APP_ID", PRO_APP_ID)
        assert self._verify(int(CUSTOMER_APP_ID))[0] == ACCOUNT_CUSTOMER

    def test_other_app_token_rejected(self, monkeypatch):
        monkeypatch.setattr(settings, "KAKAO_APP_ID", CUSTOMER_APP_ID)
        monkeypatch.setattr(settings, "KAKAO_PRO_APP_ID", PRO_APP_ID)
        with pytest.raises(HTTPException) as exc:
            self._verify(333333)
        assert exc.value.status_code == 401

    def test_pro_unset_keeps_customer_behavior(self, monkeypatch):
        """Pro 앱 ID 미설정이면 기존 검증 그대로 (Pro 토큰도 남의 앱 토큰으로 거절)"""
        monkeypatch.setattr(settings, "KAKAO_APP_ID", CUSTOMER_APP_ID)
        monkeypatch.setattr(settings, "KAKAO_PRO_APP_ID", "")
        assert self._verify(int(CUSTOMER_APP_ID))[0] == ACCOUNT_CUSTOMER
        with pytest.raises(HTTPException):
            self._verify(int(PRO_APP_ID))

    def test_same_id_for_both_apps_never_makes_pro(self, monkeypatch):
        """설정 실수(두 값이 같음)로 손님 전원이 Pro 가 되면 안 된다"""
        monkeypatch.setattr(settings, "KAKAO_APP_ID", CUSTOMER_APP_ID)
        monkeypatch.setattr(settings, "KAKAO_PRO_APP_ID", CUSTOMER_APP_ID)
        assert self._verify(int(CUSTOMER_APP_ID))[0] == ACCOUNT_CUSTOMER

    def test_nothing_configured_skips_call(self, monkeypatch):
        monkeypatch.setattr(settings, "KAKAO_APP_ID", "")
        monkeypatch.setattr(settings, "KAKAO_PRO_APP_ID", "")
        account_type, client = self._verify(1)
        assert account_type == ACCOUNT_CUSTOMER
        client.get.assert_not_called()


class TestAccountTypeHelpers:
    def test_unknown_values_are_customer(self):
        assert account_type_of_user({}) == ACCOUNT_CUSTOMER
        assert account_type_of_user({"account_type": "admin"}) == ACCOUNT_CUSTOMER
        assert account_type_of_user(None) == ACCOUNT_CUSTOMER
        assert account_type_of_user(PRO_USER) == ACCOUNT_PRO
        assert account_type_of_payload({"sub": "x"}) == ACCOUNT_CUSTOMER

    def test_customer_token_has_no_claim(self):
        payload = decode_token(create_access_token("u1"))
        assert ACCOUNT_TYPE_CLAIM not in payload
        payload = decode_token(create_access_token("u1", 1, ACCOUNT_CUSTOMER))
        assert ACCOUNT_TYPE_CLAIM not in payload

    def test_pro_token_has_claim(self):
        payload = decode_token(create_access_token("u1", 1, ACCOUNT_PRO))
        assert payload[ACCOUNT_TYPE_CLAIM] == ACCOUNT_PRO
        refresh = decode_token(
            create_refresh_token("u1", 1, ACCOUNT_PRO), TOKEN_TYPE_REFRESH
        )
        assert refresh[ACCOUNT_TYPE_CLAIM] == ACCOUNT_PRO

    def test_dependency(self):
        from fastapi.security import HTTPAuthorizationCredentials

        def creds(token):
            return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        run = asyncio.run
        assert run(get_account_type(None)) == ACCOUNT_CUSTOMER
        assert run(get_account_type(creds("broken"))) == ACCOUNT_CUSTOMER
        assert (
            run(get_account_type(creds(create_access_token("u1", 1, ACCOUNT_PRO))))
            == ACCOUNT_PRO
        )
        # 리프레시 토큰은 액세스 토큰 자리에서 인정하지 않는다
        refresh = create_refresh_token("u1", 1, ACCOUNT_PRO)
        assert run(get_account_type(creds(refresh))) == ACCOUNT_CUSTOMER


# ========== S1. 로그인 / 갱신 ==========


def _kakao_service(account_type: str, kakao_id: str = "900001"):
    service = MagicMock()
    service.verify_access_token = AsyncMock(
        return_value={
            "kakao_id": kakao_id,
            "nickname": "디자이너",
            "email": None,
            "account_type": account_type,
        }
    )
    return service


@pytest.mark.usefixtures("no_rate_limits")
class TestProLogin:
    def test_new_pro_user(self, client):
        repo = MagicMock()
        repo.get_by_kakao_id.return_value = None
        repo.create.return_value = dict(PRO_USER, credits=0)
        credit = MagicMock()
        credit.grant_with_claim.return_value = settings.PRO_SIGNUP_BONUS_CREDITS

        with patch(
            "api.endpoints.auth.get_kakao_auth_service",
            return_value=_kakao_service(ACCOUNT_PRO),
        ), patch("api.endpoints.auth.get_user_repository", return_value=repo), patch(
            "api.endpoints.auth.get_credit_service", return_value=credit
        ), patch(
            "api.endpoints.auth._signup_arm"
        ) as signup_arm, patch(
            "api.endpoints.auth.recently_withdrawn", return_value=False
        ), patch(
            "api.endpoints.auth.log_structured"
        ) as log:
            response = client.post(
                "/api/auth/kakao",
                # 앱이 device_id 를 보내도 Pro 는 무료 정책 실험에 등록하지 않는다
                json={"kakao_access_token": "t", "device_id": "a1b2c3d4e5f60718"},
            )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["is_new_user"] is True
        assert body["user"]["credits"] == settings.PRO_SIGNUP_BONUS_CREDITS
        # 회원 행에 계정 유형 저장 + 체험 수량
        kwargs = repo.create.call_args.kwargs
        assert kwargs["account_type"] == ACCOUNT_PRO
        assert kwargs["signup_bonus_pending"] == settings.PRO_SIGNUP_BONUS_CREDITS
        assert "free_policy_arm" not in kwargs
        signup_arm.assert_not_called()
        assert (
            credit.grant_with_claim.call_args.args[2]
            == settings.PRO_SIGNUP_BONUS_CREDITS
        )
        # 토큰 두 개 모두 Pro 클레임
        assert decode_token(body["access_token"])[ACCOUNT_TYPE_CLAIM] == ACCOUNT_PRO
        assert (
            decode_token(body["refresh_token"], TOKEN_TYPE_REFRESH)[ACCOUNT_TYPE_CLAIM]
            == ACCOUNT_PRO
        )
        signup = [c.args[1] for c in log.call_args_list if c.args[0] == "signup"]
        assert signup[0]["account_type"] == ACCOUNT_PRO

    def test_existing_pro_user(self, client):
        repo = MagicMock()
        repo.get_by_kakao_id.return_value = dict(PRO_USER)
        with patch(
            "api.endpoints.auth.get_kakao_auth_service",
            return_value=_kakao_service(ACCOUNT_PRO),
        ), patch("api.endpoints.auth.get_user_repository", return_value=repo):
            response = client.post("/api/auth/kakao", json={"kakao_access_token": "t"})

        assert response.status_code == 200
        assert (
            decode_token(response.json()["access_token"])[ACCOUNT_TYPE_CLAIM]
            == ACCOUNT_PRO
        )
        # 응답의 user 모양은 손님과 같다 (계정 유형 필드를 새로 넣지 않는다)
        assert "account_type" not in response.json()["user"]

    @pytest.mark.parametrize(
        "login_type, stored_user",
        [(ACCOUNT_PRO, CUSTOMER_USER), (ACCOUNT_CUSTOMER, PRO_USER)],
    )
    def test_account_type_mismatch_rejected(self, client, login_type, stored_user):
        repo = MagicMock()
        repo.get_by_kakao_id.return_value = dict(stored_user)
        with patch(
            "api.endpoints.auth.get_kakao_auth_service",
            return_value=_kakao_service(login_type),
        ), patch("api.endpoints.auth.get_user_repository", return_value=repo):
            response = client.post("/api/auth/kakao", json={"kakao_access_token": "t"})

        assert response.status_code == 403
        assert response.json()["error"] == "account_type_mismatch"
        assert "access_token" not in response.json()
        repo.update_last_login.assert_not_called()

    def test_customer_login_tokens_unchanged(self, client):
        repo = MagicMock()
        repo.get_by_kakao_id.return_value = dict(CUSTOMER_USER)
        with patch(
            "api.endpoints.auth.get_kakao_auth_service",
            return_value=_kakao_service(ACCOUNT_CUSTOMER, "900002"),
        ), patch("api.endpoints.auth.get_user_repository", return_value=repo):
            response = client.post("/api/auth/kakao", json={"kakao_access_token": "t"})

        assert response.status_code == 200
        body = response.json()
        assert set(body) == {
            "access_token",
            "refresh_token",
            "token_type",
            "is_new_user",
            "user",
        }
        assert ACCOUNT_TYPE_CLAIM not in decode_token(body["access_token"])
        assert ACCOUNT_TYPE_CLAIM not in decode_token(
            body["refresh_token"], TOKEN_TYPE_REFRESH
        )

    def test_new_customer_create_call_unchanged(self, client):
        """손님 가입은 account_type 을 넘기지 않는다 (회원 행 모양 그대로)"""
        repo = MagicMock()
        repo.get_by_kakao_id.return_value = None
        repo.create.return_value = dict(CUSTOMER_USER, credits=0)
        credit = MagicMock()
        credit.grant_with_claim.return_value = 5
        with patch(
            "api.endpoints.auth.get_kakao_auth_service",
            return_value=_kakao_service(ACCOUNT_CUSTOMER, "900002"),
        ), patch("api.endpoints.auth.get_user_repository", return_value=repo), patch(
            "api.endpoints.auth.get_credit_service", return_value=credit
        ):
            response = client.post("/api/auth/kakao", json={"kakao_access_token": "t"})

        assert response.status_code == 200
        kwargs = repo.create.call_args.kwargs
        assert "account_type" not in kwargs
        assert kwargs["signup_bonus_pending"] == settings.SIGNUP_BONUS_CREDITS


class TestRefreshKeepsAccountType:
    def _refresh(self, client, user, refresh_token):
        repo = MagicMock()
        repo.get_by_id_consistent.return_value = dict(user)
        with patch("api.endpoints.auth.get_user_repository", return_value=repo):
            return client.post(
                "/api/auth/refresh", json={"refresh_token": refresh_token}
            )

    def test_pro_stays_pro_after_refresh(self, client):
        # 클레임 없는 예전 리프레시 토큰이어도 회원 행 기준으로 Pro 가 유지된다
        response = self._refresh(client, PRO_USER, create_refresh_token("pro-user-id"))
        assert response.status_code == 200
        assert (
            decode_token(response.json()["access_token"])[ACCOUNT_TYPE_CLAIM]
            == ACCOUNT_PRO
        )

    def test_customer_refresh_has_no_claim(self, client):
        response = self._refresh(
            client, CUSTOMER_USER, create_refresh_token("customer-user-id")
        )
        assert response.status_code == 200
        assert set(response.json()) == {"access_token", "refresh_token", "token_type"}
        assert ACCOUNT_TYPE_CLAIM not in decode_token(response.json()["access_token"])


class TestUserRepositoryCreate:
    def _create(self, **kwargs):
        from database.user_repository import UserRepository

        repo = UserRepository()
        table = MagicMock()
        table.name = "hairme-users"
        repo._table = table
        with patch.object(UserRepository, "table", table):
            repo.create(kakao_id="1", nickname="n", **kwargs)
        items = table.meta.client.transact_write_items.call_args.kwargs["TransactItems"]
        return items[1]["Put"]["Item"]

    def test_pro_item_has_account_type(self):
        assert self._create(account_type=ACCOUNT_PRO)["account_type"] == ACCOUNT_PRO

    def test_customer_item_unchanged(self):
        assert "account_type" not in self._create()
        assert "account_type" not in self._create(account_type=ACCOUNT_CUSTOMER)


# ========== S2. 동의 변경 ==========


class TestConsent:
    def test_pro_cannot_change_consent(self, client):
        repo = MagicMock()
        with patch("api.endpoints.auth.get_user_repository", return_value=repo):
            response = client.patch(
                "/api/auth/me/consent",
                json={"training_consent": True},
                headers=_pro_auth(),
            )
        assert response.status_code == 403
        assert response.json()["error"] == "pro_consent_not_allowed"
        repo.set_training_consent.assert_not_called()

    def test_customer_consent_unchanged(self, client):
        repo = MagicMock()
        with patch("api.endpoints.auth.get_user_repository", return_value=repo):
            response = client.patch(
                "/api/auth/me/consent",
                json={"training_consent": True},
                headers=_customer_auth(),
            )
        assert response.status_code == 200
        assert response.json() == {"success": True, "training_consent": True}
        repo.set_training_consent.assert_called_once_with("customer-user-id", True)


# ========== S2·S4. 합성 ==========

CACHE_KEY = "c" * 64


@pytest.fixture
def synthesis_env():
    storage = MagicMock()
    storage.enabled = True
    storage.build_cache_key.return_value = CACHE_KEY
    storage.get_cached_result.return_value = None
    storage.save_user_result.return_value = "https://s3/result.png"
    storage.save_pro_result.return_value = {
        "result_url": "https://s3/result.png",
        "share_token": "T" * 22,
        "created_at": datetime(2026, 11, 10, 3, 0, tzinfo=timezone.utc),
    }

    service = MagicMock()
    service.synthesize_hairstyle.return_value = {
        "success": True,
        "image_base64": "aW1n",
        "image_format": "png",
        "message": "적용되었습니다.",
        "api_calls": 1,
    }
    credit = MagicMock()
    credit.consume_for_synthesis.return_value = (9, "free")
    credit.get_wallet.return_value = {"paid_balance": 0}
    user_repo = MagicMock()
    user_repo.get_by_id.return_value = dict(CUSTOMER_USER)
    products = MagicMock()
    products.get_recommendations.return_value = [{"product_id": "p1"}]
    products.disclosure = "대가성 문구"

    def finalize(result, tier, watermark_text=None):
        return f"{result['image_base64']}|{watermark_text}", result["image_format"]

    with patch("api.endpoints.synthesis.record_api_calls"), patch(
        "api.endpoints.synthesis.daily_budget_exceeded", return_value=False
    ), patch(
        "api.endpoints.synthesis.acquire_synthesis_lock",
        return_value=(True, lambda: None),
    ), patch(
        "api.endpoints.synthesis.get_photo_storage_service", return_value=storage
    ), patch(
        "api.endpoints.synthesis.get_credit_service", return_value=credit
    ), patch(
        "api.endpoints.synthesis.get_synthesis_service", return_value=service
    ), patch(
        "api.endpoints.synthesis.get_user_repository", return_value=user_repo
    ), patch(
        "api.endpoints.synthesis.get_product_recommendation_service",
        return_value=products,
    ), patch(
        "api.endpoints.synthesis.finalize_image", side_effect=finalize
    ) as finalize_mock, patch(
        "api.endpoints.synthesis.record_style_signal"
    ) as record, patch(
        "api.endpoints.synthesis.log_structured"
    ) as log:
        yield {
            "storage": storage,
            "user_repo": user_repo,
            "record": record,
            "log": log,
            "finalize": finalize_mock,
        }


def _synthesize(client, headers):
    return client.post(
        "/api/v2/synthesize",
        files={"file": ("face.png", _png_bytes(), "image/png")},
        data={
            "hairstyle_name": "레이어드 컷",
            "gender": "female",
            # 손님 앱이 보내는 추천 카드 정보 - Pro 는 무시돼야 한다
            "analysis_id": "11111111-2222-3333-4444-555555555555",
            "rec_rank": "1",
            "hairstyle_id": "3",
        },
        headers=headers,
    )


def _events(log, name):
    return [c.args[1] for c in log.call_args_list if c.args[0] == name]


@pytest.mark.usefixtures("no_rate_limits")
class TestProSynthesis:
    def test_pro_privacy_and_response(self, client, synthesis_env):
        env = synthesis_env
        response = _synthesize(client, _pro_auth())
        assert response.status_code == 200, response.text
        body = response.json()

        storage = env["storage"]
        # 공용 캐시: 읽지도 쓰지도 않는다
        storage.get_cached_result.assert_not_called()
        storage.save_cached_result.assert_not_called()
        # 원본 사진: 저장하지 않는다 (동의 조회도 하지 않는다)
        storage.save_original_photo.assert_not_called()
        env["user_repo"].get_by_id.assert_not_called()
        # 결과: pro/results/ 경로로 저장 (손님용 저장 함수는 부르지 않는다)
        storage.save_pro_result.assert_called_once_with(
            "pro-user-id", "aW1n|Hair Consultant", "png", "레이어드 컷"
        )
        storage.save_user_result.assert_not_called()
        assert body["result_url"] == "https://s3/result.png"
        # 학습 신호: 추천 카드 정보를 보내도 기록하지 않는다
        env["record"].assert_not_called()
        # 제휴 상품: 응답에 키 자체가 없다
        assert "recommended_products" not in body
        assert "disclosure" not in body
        # 워터마크: Pro 브랜드
        assert env["finalize"].call_args.args[2] == settings.PRO_RESULT_WATERMARK_TEXT
        # 로그: 계정 유형
        assert _events(env["log"], "synthesis_start")[0]["account_type"] == "pro"
        success = _events(env["log"], "synthesis_success")[0]
        assert success["account_type"] == "pro"
        assert success["analysis_id"] is None

    def test_pro_ignores_existing_cache_entry(self, client, synthesis_env):
        """같은 사진·스타일이 캐시에 있어도 Pro 는 새로 합성하고 차감된다"""
        synthesis_env["storage"].get_cached_result.return_value = {
            "image_base64": "Y2FjaGVk",
            "image_format": "png",
        }
        response = _synthesize(client, _pro_auth())
        assert response.json()["cached"] is False
        assert response.json()["quota"]["mode"] == "credits"

    def test_customer_response_and_calls_unchanged(self, client, synthesis_env):
        env = synthesis_env
        response = _synthesize(client, _customer_auth())
        assert response.status_code == 200
        body = response.json()
        body.pop("processing_time")
        assert body == {
            "success": True,
            "image_base64": "aW1n|None",
            "image_format": "png",
            "message": "적용되었습니다.",
            "cached": False,
            "result_url": "https://s3/result.png",
            "quota": {"mode": "credits", "balance": 9, "tier": "free"},
            "quality": {
                "tier": "free",
                "image_size": settings.FREE_IMAGE_SIZE or None,
                "watermarked": settings.FREE_RESULT_WATERMARK,
            },
            "is_preview": True,
            "result_id": CACHE_KEY[:16],
            "recommended_products": [{"product_id": "p1"}],
            "disclosure": "대가성 문구",
        }
        # 응답 키 순서도 그대로 (제휴 상품 두 키가 끝)
        assert list(response.json())[-2:] == ["recommended_products", "disclosure"]
        storage = env["storage"]
        storage.get_cached_result.assert_called_once()
        storage.save_cached_result.assert_called_once()
        # 기존과 같은 인자 (pro 인자를 넘기지 않는다)
        storage.save_user_result.assert_called_once_with(
            "customer-user-id", "aW1n|None", "png", "레이어드 컷"
        )
        storage.save_pro_result.assert_not_called()
        # 동의한 손님은 원본 저장 (기존 동작)
        storage.save_original_photo.assert_called_once()
        env["record"].assert_called_once()
        assert env["finalize"].call_args.args == (
            env["finalize"].call_args.args[0],
            "free",
        )
        for name in ("synthesis_start", "synthesis_success"):
            assert "account_type" not in _events(env["log"], name)[0]

    def test_customer_cache_hit_unchanged(self, client, synthesis_env):
        synthesis_env["storage"].get_cached_result.return_value = {
            "image_base64": "Y2FjaGVk",
            "image_format": "png",
        }
        response = _synthesize(client, _customer_auth())
        body = response.json()
        assert body["cached"] is True
        assert body["quota"] is None
        assert body["recommended_products"] == [{"product_id": "p1"}]


# ========== S2. 결과 목록 / 저장 경로 ==========


class TestResultsPrefix:
    def _service(self, monkeypatch):
        from services.photo_storage_service import PhotoStorageService

        monkeypatch.setenv("PHOTO_S3_BUCKET", "test-bucket")
        monkeypatch.setattr(settings, "PHOTO_S3_BUCKET", "test-bucket")
        svc = PhotoStorageService()
        svc._client = MagicMock()
        svc._client.generate_presigned_url.return_value = "https://s3/presigned"
        svc._client.list_objects_v2.return_value = {
            "Contents": [],
            "IsTruncated": False,
        }
        return svc

    def test_save_paths(self, monkeypatch):
        svc = self._service(monkeypatch)
        svc.save_user_result("u1", "aW1n", "png", pro=True)
        svc.save_user_result("u1", "aW1n", "png")
        keys = [c.kwargs["Key"] for c in svc._client.put_object.call_args_list]
        # Pro: 결과 + 공유 주소 연결 정보 (둘 다 pro/results/ 아래)
        assert keys[0].startswith("pro/results/u1/")
        assert keys[1].startswith("pro/results/_share/")
        assert keys[2].startswith("results/u1/")
        assert len(keys) == 3

    def test_list_paths(self, monkeypatch):
        svc = self._service(monkeypatch)
        svc.list_user_results("u1", pro=True)
        svc.list_user_results("u1")
        prefixes = [
            c.kwargs["Prefix"] for c in svc._client.list_objects_v2.call_args_list
        ]
        assert prefixes == ["pro/results/u1/", "results/u1/"]

    def test_endpoint_routes_by_account_type(self, client):
        service = MagicMock()
        service.list_user_results.return_value = {"items": [], "next_token": None}
        with patch(
            "api.endpoints.results.get_photo_storage_service", return_value=service
        ):
            assert client.get("/api/me/results", headers=_pro_auth()).status_code == 200
            assert (
                client.get("/api/me/results", headers=_customer_auth()).status_code
                == 200
            )
        pro_call, customer_call = service.list_user_results.call_args_list
        assert pro_call.args == ("pro-user-id", 20, None)
        assert pro_call.kwargs == {"pro": True}
        assert customer_call.args == ("customer-user-id", 20, None)
        assert customer_call.kwargs == {}


class TestLifecycleScript:
    def test_adds_rule_and_keeps_existing(self):
        from scripts.apply_pro_results_lifecycle import DAYS, PREFIX, merged_rules

        existing = [
            {"ID": "expire-results", "Filter": {"Prefix": "results/"}},
            {"ID": "expire-pro-results", "Filter": {"Prefix": "old/"}},
        ]
        rules = merged_rules(existing)
        assert rules[0] == existing[0]
        assert len(rules) == 2
        assert rules[1]["Filter"] == {"Prefix": "pro/results/"}
        assert rules[1]["Expiration"] == {"Days": 7}
        assert PREFIX == "pro/results/" and DAYS == 7
        # 손님용 results/ 규칙에 Pro 경로가 걸리지 않는다
        assert not PREFIX.startswith("results/")


# ========== S2. 이벤트 / 분석 / 염색 ==========


@pytest.mark.usefixtures("no_rate_limits")
class TestEvents:
    BODY = {
        "analysis_id": "11111111-2222-3333-4444-555555555555",
        "style_name": "레이어드 컷",
        "action": "save",
    }

    def test_pro_event_not_recorded(self, client):
        with patch("api.endpoints.events.record_style_signal") as record:
            response = client.post(
                "/api/events/style-action", json=self.BODY, headers=_pro_auth()
            )
        assert response.status_code == 200
        assert response.json() == {"success": True}
        record.assert_not_called()

    def test_customer_event_recorded(self, client):
        with patch("api.endpoints.events.record_style_signal") as record:
            response = client.post(
                "/api/events/style-action", json=self.BODY, headers=_customer_auth()
            )
        assert response.json() == {"success": True}
        record.assert_called_once()


def _analyze(headers):
    from core.dependencies import get_face_detection_service, get_hybrid_service

    features = MagicMock(
        face_shape="계란형",
        skin_tone="봄웜",
        face_features=[1.4, 120.0, 130.0, 110.0, 0.92, 0.85],
        skin_features=[45.0, 15.0],
        confidence=0.9,
    )
    detector = MagicMock()
    detector.detect_face.return_value = {"has_face": True, "features": features}
    recommender = MagicMock()
    recommender.recommend.return_value = {
        "recommendations": [
            {"style_name": "레이어드 컷", "reason": "r", "score": 80.0, "source": "ml"}
        ]
    }
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color="white").save(buffer, format="JPEG")

    app.dependency_overrides[get_face_detection_service] = lambda: detector
    app.dependency_overrides[get_hybrid_service] = lambda: recommender
    try:
        with patch(
            "services.trending_style_service.get_trending_style_service"
        ) as trending, patch(
            "api.endpoints.analyze.save_to_database", return_value="analysis-1"
        ) as save, patch(
            "api.endpoints.analyze.limiter.enabled", False
        ), TestClient(
            app
        ) as test_client:
            trending.return_value.pick_trending.return_value = []
            response = test_client.post(
                "/api/v2/analyze-hybrid",
                files={"file": ("face.jpg", buffer.getvalue(), "image/jpeg")},
                data={"gender": "male"},
                headers=headers,
            )
    finally:
        app.dependency_overrides.pop(get_face_detection_service, None)
        app.dependency_overrides.pop(get_hybrid_service, None)
    return response, save


class TestAnalyze:
    def test_pro_analysis_not_saved(self):
        response, save = _analyze(_pro_auth())
        assert response.status_code == 200, response.text
        save.assert_not_called()
        assert response.json()["analysis_id"] is None

    def test_customer_analysis_saved(self):
        response, save = _analyze(_customer_auth())
        assert response.status_code == 200, response.text
        save.assert_called_once()
        assert response.json()["analysis_id"] == "analysis-1"


@pytest.mark.usefixtures("no_rate_limits")
class TestHairColor:
    def test_pro_rejected_before_charge(self, client):
        with patch("api.endpoints.hair_color.charge_synthesis_quota") as charge:
            response = client.post(
                "/api/hair-color/synthesize",
                files={"file": ("face.png", _png_bytes(), "image/png")},
                data={"color_name": "밀크브라운"},
                headers=_pro_auth(),
            )
        assert response.status_code == 403
        assert response.json()["error"] == "pro_not_supported"
        charge.assert_not_called()


# ========== S3. 결제 ==========


@pytest.fixture
def pro_billing(monkeypatch):
    monkeypatch.setattr(settings, "PLAY_PRO_PACKAGE_NAME", PRO_PACKAGE)
    monkeypatch.setenv("PLAY_PRO_PACKAGE_NAME", PRO_PACKAGE)
    play = MagicMock()
    play.pro_package_name = PRO_PACKAGE
    play.verify_product_purchase.return_value = {
        "order_id": "GPA.1",
        "acknowledgement_state": 0,
        "obfuscated_account_id": None,
    }
    credit = MagicMock()
    credit.grant_with_claim.return_value = 40
    credit.get_wallet.return_value = {
        "balance": 10,
        "paid_balance": 0,
        "first_purchase_at": None,
    }
    with patch(
        "api.endpoints.credits.get_play_billing_service", return_value=play
    ), patch("api.endpoints.credits.get_credit_service", return_value=credit):
        yield {"play": play, "credit": credit}


def _purchase(client, headers, product_id):
    return client.post(
        "/api/credits/purchase",
        json={"product_id": product_id, "purchase_token": "tok-1"},
        headers=headers,
    )


@pytest.mark.usefixtures("no_rate_limits")
class TestProBilling:
    def test_products_are_pro_only(self, client, pro_billing):
        response = client.get("/api/credits/products", headers=_pro_auth())
        assert response.status_code == 200
        body = response.json()
        assert [p["product_id"] for p in body["products"]] == list(
            settings.PRO_CREDIT_PRODUCTS
        )
        assert body["paywall_variant"] is None
        assert body["balance"] == 10

    def test_customer_products_unchanged(self, client, pro_billing):
        pro_billing["credit"].get_claim.return_value = None
        response = client.get("/api/credits/products", headers=_customer_auth())
        ids = [p["product_id"] for p in response.json()["products"]]
        assert ids == list(settings.CREDIT_PRODUCTS)
        assert response.json()["paywall_variant"] in ("recommended", "secondary")

    def test_pro_purchase_uses_pro_package(self, client, pro_billing):
        product_id = next(iter(settings.PRO_CREDIT_PRODUCTS))
        response = _purchase(client, _pro_auth(), product_id)
        assert response.status_code == 200, response.text
        assert response.json()["granted"] == settings.PRO_CREDIT_PRODUCTS[product_id]
        play = pro_billing["play"]
        assert play.verify_product_purchase.call_args.kwargs == {
            "package_name": PRO_PACKAGE
        }
        assert play.acknowledge_product_purchase.call_args.kwargs == {
            "package_name": PRO_PACKAGE
        }
        # 유료분으로 지급 → 1K, 워터마크 없음
        assert pro_billing["credit"].grant_with_claim.call_args.kwargs["paid"] is True

    def test_customer_purchase_call_unchanged(self, client, pro_billing):
        response = _purchase(client, _customer_auth(), "credits_10")
        assert response.status_code == 200, response.text
        play = pro_billing["play"]
        assert play.verify_product_purchase.call_args.args == ("credits_10", "tok-1")
        assert play.verify_product_purchase.call_args.kwargs == {}
        assert play.acknowledge_product_purchase.call_args.kwargs == {}

    def test_pro_cannot_buy_customer_product(self, client, pro_billing):
        response = _purchase(client, _pro_auth(), "credits_10")
        assert response.status_code == 400
        pro_billing["play"].verify_product_purchase.assert_not_called()

    def test_customer_cannot_buy_pro_product(self, client, pro_billing):
        response = _purchase(
            client, _customer_auth(), next(iter(settings.PRO_CREDIT_PRODUCTS))
        )
        assert response.status_code == 400
        pro_billing["play"].verify_product_purchase.assert_not_called()

    def test_pro_package_missing_is_503(self, client, pro_billing):
        pro_billing["play"].pro_package_name = ""
        response = _purchase(
            client, _pro_auth(), next(iter(settings.PRO_CREDIT_PRODUCTS))
        )
        assert response.status_code == 503
        pro_billing["play"].verify_product_purchase.assert_not_called()

    def test_product_ids_do_not_overlap(self):
        assert not set(settings.PRO_CREDIT_PRODUCTS) & set(settings.CREDIT_PRODUCTS)


class TestPlayBillingPackage:
    def test_package_name_override(self, monkeypatch):
        from services.play_billing_service import PlayBillingService

        monkeypatch.setenv("PLAY_PACKAGE_NAME", CUSTOMER_PACKAGE)
        svc = PlayBillingService()
        session = MagicMock()
        session.get.return_value.status_code = 200
        session.get.return_value.json.return_value = {"purchaseState": 0}
        session.post.return_value.status_code = 200
        svc._session = session

        svc.verify_product_purchase("p", "t", package_name=PRO_PACKAGE)
        svc.verify_product_purchase("p", "t")
        svc.acknowledge_product_purchase("p", "t", package_name=PRO_PACKAGE)
        svc.list_voided_purchases(0, package_name=PRO_PACKAGE)

        get_urls = [c.args[0] for c in session.get.call_args_list]
        assert f"/applications/{PRO_PACKAGE}/" in get_urls[0]
        assert f"/applications/{CUSTOMER_PACKAGE}/" in get_urls[1]
        assert f"/applications/{PRO_PACKAGE}/" in get_urls[2]
        assert f"/applications/{PRO_PACKAGE}/" in session.post.call_args.args[0]


class TestVoidReclaimBothPackages:
    def _run(self, pro_package):
        from services.play_void_reclaim_service import PlayVoidReclaimService

        play = MagicMock()
        play.package_name = CUSTOMER_PACKAGE
        play.pro_package_name = pro_package
        play.list_voided_purchases.return_value = {"voidedPurchases": []}
        PlayVoidReclaimService(play, MagicMock()).run()
        return play.list_voided_purchases.call_args_list

    def test_pro_package_is_polled(self):
        calls = self._run(PRO_PACKAGE)
        assert len(calls) == 2
        assert calls[0].kwargs == {"page_token": None}
        assert calls[1].kwargs == {"page_token": None, "package_name": PRO_PACKAGE}

    def test_without_pro_package_unchanged(self):
        calls = self._run("")
        assert len(calls) == 1
        assert calls[0].kwargs == {"page_token": None}

    def test_pro_product_amount_resolved(self):
        from services.play_void_reclaim_service import PlayVoidReclaimService

        product_id, amount = next(iter(settings.PRO_CREDIT_PRODUCTS.items()))
        credit = MagicMock()
        credit.get_claim.return_value = {
            "claimed_by": "pro-user-id",
            "detail": {"product_id": product_id},
        }
        resolved = PlayVoidReclaimService(MagicMock(), credit)._resolve_grant("tok")
        assert resolved == ("pro-user-id", amount)
        credit.find_ledger_entry.assert_not_called()
