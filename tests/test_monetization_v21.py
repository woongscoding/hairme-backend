"""수익화 v2.1 Phase 1 서버 기능 테스트

- 무료 정책 실험 코호트 (services/free_policy_service.py)
- 실험군 비회원 평생 1회 / 대조군 일일 3회 (core/quota.py, /api/usage)
- 가입 보너스 실험군 3회 / 대조군 5회 (/api/auth/kakao)
- 보상형 광고: 크레딧 vs 원가 0원 기능 해제 (/api/credits/reward-callback, /api/rewards/unlocks)
- 무료 결과 워터마크, Gemini image_size 전달
- AI 생성 이미지 신고 (/api/results/report)

DynamoDB 조건식이 걸린 부분은 moto 위에서 실제 테이블로 돈다.
"""

import base64
import io
import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
from PIL import Image

import services.usage_limit_service as usage_module
from config.settings import settings
from core.jwt_auth import create_access_token
from main import app
from services import free_policy_service as fp
from services.result_watermark import apply_watermark
from services.usage_limit_service import UsageLimitService

REGION = "ap-northeast-2"
PAST = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
FUTURE = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()


def _device_in(arm_wanted: str) -> str:
    """해시 버킷이 원하는 쪽에 떨어지는 device_id 를 찾는다"""
    for i in range(1000):
        device_id = f"device-test-{i:04d}"
        in_treatment = (
            fp.hash_bucket(device_id) < settings.FREE_POLICY_TREATMENT_PERCENT
        )
        if (arm_wanted == fp.ARM_TREATMENT) == in_treatment:
            return device_id
    raise AssertionError("device_id 를 찾지 못함")


@pytest.fixture(autouse=True)
def no_rate_limit(monkeypatch):
    """분당 호출 제한은 테스트 간에 공유되므로 끈다 (기존 합성 테스트와 같은 방식)"""
    from api.endpoints import auth, credits, results, synthesis

    for module in (synthesis, credits, results, auth):
        monkeypatch.setattr(module.limiter, "enabled", False)


@pytest.fixture
def experiment_on(monkeypatch):
    monkeypatch.setattr(settings, "FREE_POLICY_EXPERIMENT_ENABLED", True)
    monkeypatch.setattr(settings, "FREE_POLICY_EXPERIMENT_START", PAST)


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (180, 140, 120)).save(buf, format="JPEG")
    return buf.getvalue()


def _png_b64(size=(128, 128), color=(30, 30, 30)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ========== 코호트 배정 ==========


class TestCohortAssignment:
    def test_disabled_experiment_grandfathers_new_devices(self, monkeypatch):
        monkeypatch.setattr(settings, "FREE_POLICY_EXPERIMENT_ENABLED", False)
        monkeypatch.setattr(settings, "FREE_POLICY_EXPERIMENT_START", PAST)
        assert (
            fp.arm_for_new_device(_device_in(fp.ARM_TREATMENT)) == fp.ARM_GRANDFATHERED
        )

    def test_before_start_grandfathers(self, monkeypatch):
        monkeypatch.setattr(settings, "FREE_POLICY_EXPERIMENT_ENABLED", True)
        monkeypatch.setattr(settings, "FREE_POLICY_EXPERIMENT_START", FUTURE)
        assert (
            fp.arm_for_new_device(_device_in(fp.ARM_TREATMENT)) == fp.ARM_GRANDFATHERED
        )

    def test_empty_or_invalid_start_grandfathers(self, monkeypatch):
        monkeypatch.setattr(settings, "FREE_POLICY_EXPERIMENT_ENABLED", True)
        for raw in ("", "not-a-date"):
            monkeypatch.setattr(settings, "FREE_POLICY_EXPERIMENT_START", raw)
            assert fp.arm_for_new_device("device-xyz-123") == fp.ARM_GRANDFATHERED

    def test_after_start_splits_by_hash(self, experiment_on):
        assert fp.arm_for_new_device(_device_in(fp.ARM_TREATMENT)) == fp.ARM_TREATMENT
        assert fp.arm_for_new_device(_device_in(fp.ARM_CONTROL)) == fp.ARM_CONTROL

    def test_split_is_roughly_half(self, experiment_on):
        arms = [fp.arm_for_new_device(f"device-split-{i}") for i in range(2000)]
        share = arms.count(fp.ARM_TREATMENT) / len(arms)
        assert 0.45 < share < 0.55

    def test_turning_off_reverts_everyone_to_control(self, monkeypatch):
        monkeypatch.setattr(settings, "FREE_POLICY_EXPERIMENT_ENABLED", False)
        assert fp.effective_arm(fp.ARM_TREATMENT) == fp.ARM_CONTROL
        assert fp.signup_bonus_for(fp.ARM_TREATMENT) == settings.SIGNUP_BONUS_CREDITS
        assert fp.reward_grants_credit(fp.ARM_TREATMENT) is True

    def test_policy_values_per_arm(self, experiment_on):
        assert fp.signup_bonus_for(fp.ARM_TREATMENT) == 3
        assert fp.signup_bonus_for(fp.ARM_GRANDFATHERED) == 5
        assert fp.signup_bonus_for(None) == 5
        assert fp.reward_grants_credit(fp.ARM_TREATMENT) is False
        assert fp.reward_grants_credit(fp.ARM_CONTROL) is True

    def test_registry_failure_falls_back_to_control(self, experiment_on):
        usage = MagicMock()
        usage.register_cohort.side_effect = RuntimeError("ddb down")
        assert (
            fp.resolve_device_arm(usage, _device_in(fp.ARM_TREATMENT)) == fp.ARM_CONTROL
        )


# ========== moto 위 비회원 합성 / 사용량 / 가입 ==========


@pytest.fixture
def usage_env():
    with mock_aws():
        ddb = boto3.resource("dynamodb", region_name=REGION)
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
        usage = UsageLimitService()
        synthesis = MagicMock()
        synthesis.synthesize_hairstyle.return_value = {
            "success": True,
            "image_base64": _png_b64(),
            "image_format": "png",
            "message": "ok",
            "api_calls": 1,
        }
        storage = MagicMock()
        storage.enabled = False
        storage.get_cached_result.return_value = None
        storage.build_cache_key.side_effect = lambda *parts: "c" * 64
        with patch.object(usage_module, "_usage_limit_service", usage), patch(
            "api.endpoints.synthesis.get_synthesis_service", return_value=synthesis
        ), patch(
            "api.endpoints.synthesis.get_photo_storage_service", return_value=storage
        ), patch(
            "api.endpoints.synthesis._safe_product_recommendations", return_value=[]
        ), TestClient(
            app
        ) as client:
            yield {"client": client, "usage": usage, "synthesis": synthesis, "ddb": ddb}


def _guest_synthesize(client, device_id):
    return client.post(
        "/api/v2/synthesize",
        files={"file": ("face.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"hairstyle_name": "리프컷", "gender": "male", "device_id": device_id},
    )


@pytest.mark.moto
class TestGuestFreePolicy:
    def test_treatment_device_gets_one_lifetime_synthesis(
        self, usage_env, experiment_on
    ):
        client = usage_env["client"]
        device_id = _device_in(fp.ARM_TREATMENT)

        first = _guest_synthesize(client, device_id)
        assert first.status_code == 200, first.text
        body = first.json()
        assert body["quota"]["policy"] == "treatment"
        assert body["quota"]["lifetime_limit"] == 1
        assert body["quota"]["remaining"] == 0
        assert body["quality"] == {
            "tier": "free",
            "image_size": "512",
            "watermarked": True,
        }
        assert body["is_preview"] is True

        second = _guest_synthesize(client, device_id)
        assert second.status_code == 429
        assert second.json()["error"] == "free_trial_used"
        # 앱은 limit_type == "device" 일 때만 결제 화면을 띄운다
        assert second.json()["limit_type"] == "device"
        assert second.json()["limit_detail"] == "lifetime"
        assert "reset_at" not in second.json()

        usage = client.get("/api/usage", params={"device_id": device_id}).json()
        assert usage == {
            "policy": "treatment",
            "experiment_arm": "treatment",
            "free_policy": "lifetime",
            "signup_bonus": settings.TREATMENT_SIGNUP_BONUS_CREDITS,
            "lifetime_limit": 1,
            "used": 1,
            "remaining": 0,
        }

    def test_treatment_failure_refunds_lifetime_quota(self, usage_env, experiment_on):
        client = usage_env["client"]
        device_id = _device_in(fp.ARM_TREATMENT)
        usage_env["synthesis"].synthesize_hairstyle.return_value = {
            "success": False,
            "image_base64": None,
            "image_format": None,
            "message": "fail",
            "api_calls": 3,
        }
        assert _guest_synthesize(client, device_id).status_code == 422
        assert usage_env["usage"].get_lifetime_count(f"lifetime#{device_id}") == 0

    def test_control_device_keeps_daily_limit(self, usage_env, experiment_on):
        client = usage_env["client"]
        device_id = _device_in(fp.ARM_CONTROL)
        for _ in range(settings.DAILY_SYNTHESIS_LIMIT):
            response = _guest_synthesize(client, device_id)
            assert response.status_code == 200
            assert response.json()["quota"]["policy"] == "control"
        blocked = _guest_synthesize(client, device_id)
        assert blocked.status_code == 429
        assert blocked.json()["limit_type"] == "device"
        assert blocked.json()["limit_detail"] == "daily"

        usage = client.get("/api/usage", params={"device_id": device_id}).json()
        assert usage["experiment_arm"] == "control"
        assert usage["free_policy"] == "daily"
        assert usage["signup_bonus"] == settings.SIGNUP_BONUS_CREDITS
        assert usage["remaining"] == 0

    def test_device_seen_before_start_stays_control(self, usage_env, monkeypatch):
        """실험 전에 등록된 기기는 실험을 켠 뒤에도 대조군"""
        client = usage_env["client"]
        device_id = _device_in(fp.ARM_TREATMENT)
        monkeypatch.setattr(settings, "FREE_POLICY_EXPERIMENT_ENABLED", False)
        assert (
            client.get("/api/usage", params={"device_id": device_id}).json()["policy"]
            == "control"
        )

        monkeypatch.setattr(settings, "FREE_POLICY_EXPERIMENT_ENABLED", True)
        monkeypatch.setattr(settings, "FREE_POLICY_EXPERIMENT_START", PAST)
        response = _guest_synthesize(client, device_id)
        assert response.json()["quota"]["policy"] == "control"

    def test_cohort_is_sticky(self, usage_env, experiment_on, monkeypatch):
        """솔트를 바꿔도(해시가 바뀌어도) 이미 배정된 기기의 코호트는 그대로"""
        device_id = _device_in(fp.ARM_TREATMENT)
        assert fp.resolve_device_arm(usage_env["usage"], device_id) == fp.ARM_TREATMENT
        monkeypatch.setattr(settings, "FREE_POLICY_SALT", "another-salt")
        assert fp.resolve_device_arm(usage_env["usage"], device_id) == fp.ARM_TREATMENT


@pytest.mark.moto
class TestSignupBonusByArm:
    def _login(self, client, device_id=None):
        kakao = MagicMock()

        async def _verify(token):
            return {"kakao_id": "k-1", "nickname": "n", "email": None}

        kakao.verify_access_token.side_effect = _verify
        repo = MagicMock()
        repo.get_by_kakao_id.return_value = None
        repo.create.side_effect = lambda **kw: {
            "user_id": "new-user",
            "credits": 0,
            "free_policy_arm": kw.get("free_policy_arm"),
        }
        credit = MagicMock()
        credit.grant.side_effect = lambda user_id, amount, reason: amount
        payload = {"kakao_access_token": "t"}
        if device_id:
            payload["device_id"] = device_id
        with patch(
            "api.endpoints.auth.get_kakao_auth_service", return_value=kakao
        ), patch("api.endpoints.auth.get_user_repository", return_value=repo), patch(
            "api.endpoints.auth.get_credit_service", return_value=credit
        ):
            response = client.post("/api/auth/kakao", json=payload)
        return response, repo, credit

    def test_treatment_device_gets_3(self, usage_env, experiment_on):
        response, repo, credit = self._login(
            usage_env["client"], _device_in(fp.ARM_TREATMENT)
        )
        assert response.status_code == 200, response.text
        assert response.json()["user"]["credits"] == 3
        assert repo.create.call_args.kwargs["free_policy_arm"] == fp.ARM_TREATMENT

    def test_control_device_gets_5(self, usage_env, experiment_on):
        response, _, _ = self._login(usage_env["client"], _device_in(fp.ARM_CONTROL))
        assert response.json()["user"]["credits"] == 5

    def test_old_app_without_device_gets_5(self, usage_env, experiment_on):
        response, repo, _ = self._login(usage_env["client"])
        assert response.json()["user"]["credits"] == 5
        assert repo.create.call_args.kwargs["free_policy_arm"] is None


# ========== 보상형 광고 ==========


@pytest.mark.moto
class TestRewardUnlock:
    def _callback(self, client, user_id, tx="tx-1"):
        ssv = MagicMock()
        ssv.verify_callback.return_value = {
            "user_id": user_id,
            "transaction_id": tx,
            "ad_unit": "unit-1",
        }
        credit = MagicMock()
        credit.try_claim_ref.return_value = True
        credit.grant.return_value = 7
        with patch(
            "api.endpoints.credits.get_admob_ssv_service", return_value=ssv
        ), patch("api.endpoints.credits.get_credit_service", return_value=credit):
            return client.get("/api/credits/reward-callback?x=1"), credit

    def test_control_member_still_gets_credit(self, usage_env):
        response, credit = self._callback(usage_env["client"], "member-1")
        assert response.json() == {
            "success": True,
            "rewarded": True,
            "reward": "credit",
        }
        credit.grant.assert_called_once()

    def test_global_switch_turns_reward_into_unlock(self, usage_env, monkeypatch):
        monkeypatch.setattr(settings, "REWARD_AD_GRANTS_CREDIT", False)
        client = usage_env["client"]
        response, credit = self._callback(client, "member-1")
        body = response.json()
        assert body["reward"] == "unlock"
        assert set(body["features"]) == set(fp.REWARD_UNLOCK_FEATURES)
        credit.grant.assert_not_called()

        status = client.get(
            "/api/rewards/unlocks",
            headers={"Authorization": f"Bearer {create_access_token('member-1')}"},
        ).json()
        assert status["unlocked"] is True
        assert status["reward_mode"] == "unlock"
        assert status["expires_at"] == body["expires_at"]

    def test_treatment_member_gets_unlock(self, usage_env, experiment_on):
        repo = MagicMock()
        repo.get_by_id.return_value = {"user_id": "m", "free_policy_arm": "treatment"}
        with patch("api.endpoints.credits.get_user_repository", return_value=repo):
            response, credit = self._callback(usage_env["client"], "m")
        assert response.json()["reward"] == "unlock"
        credit.grant.assert_not_called()

    def test_guest_device_subject_unlocks(self, usage_env):
        client = usage_env["client"]
        response, credit = self._callback(client, "device:device-guest-0001")
        assert response.json()["reward"] == "unlock"
        credit.grant.assert_not_called()

        status = client.get(
            "/api/rewards/unlocks", params={"device_id": "device-guest-0001"}
        ).json()
        assert status["unlocked"] is True
        assert status["reward_mode"] == "unlock"

    def test_invalid_device_subject_is_200_without_reward(self, usage_env):
        response, _ = self._callback(usage_env["client"], "device:bad#id")
        assert response.status_code == 200
        assert response.json()["reason"] == "invalid_subject"

    def test_unlocks_without_identity_is_400(self, usage_env):
        assert usage_env["client"].get("/api/rewards/unlocks").status_code == 400


# ========== 상품 목록 ==========


class TestProductCatalog:
    def test_catalog_lists_products_without_credits_100(self):
        credit = MagicMock()
        credit.get_wallet.return_value = {
            "balance": 2,
            "paid_balance": 0,
            "first_purchase_at": None,
        }
        credit.get_claim.return_value = None
        with patch(
            "api.endpoints.credits.get_credit_service", return_value=credit
        ), TestClient(app) as client:
            response = client.get(
                "/api/credits/products",
                headers={"Authorization": f"Bearer {create_access_token('u-cat')}"},
            )
        assert response.status_code == 200
        body = response.json()
        ids = [p["product_id"] for p in body["products"]]
        assert ids == ["starter_5", "credits_10", "credits_30"]
        assert body["ad_free"] is False
        assert body["paywall_variant"] in ("recommended", "secondary")
        credits_30 = body["products"][2]
        assert credits_30["badge"] == (
            "recommended" if body["paywall_variant"] == "recommended" else None
        )
        assert body["products"][1]["default"] is True


# ========== 워터마크 / image_size ==========


class TestWatermark:
    def test_watermark_changes_pixels_and_keeps_size(self):
        original = _png_b64((256, 256))
        marked, fmt = apply_watermark(original, "png", "HairMe")
        assert fmt == "png"
        assert marked != original
        image = Image.open(io.BytesIO(base64.b64decode(marked)))
        assert image.size == (256, 256)

    def test_jpeg_stays_jpeg(self):
        buf = io.BytesIO()
        Image.new("RGB", (200, 200), (10, 10, 10)).save(buf, format="JPEG")
        marked, fmt = apply_watermark(
            base64.b64encode(buf.getvalue()).decode(), "jpeg", "HairMe"
        )
        assert fmt == "jpeg"
        assert base64.b64decode(marked)[:3] == b"\xff\xd8\xff"

    def test_broken_input_returns_original(self):
        assert apply_watermark("not-base64-image", "png", "HairMe") == (
            "not-base64-image",
            "png",
        )


class TestImageSizeConfig:
    def _service(self):
        from services.hairstyle_synthesis_service import HairstyleSynthesisService

        service = HairstyleSynthesisService()
        part = MagicMock()
        part.inline_data = MagicMock(
            data=base64.b64decode(_png_b64()), mime_type="image/png"
        )
        part.text = None
        response = MagicMock()
        response.candidates = [MagicMock(content=MagicMock(parts=[part]))]
        service._client = MagicMock()
        service._client.models.generate_content.return_value = response
        return service

    def _jpeg(self):
        return _jpeg_bytes()

    def test_image_size_is_passed(self):
        service = self._service()
        service.synthesize_hairstyle(self._jpeg(), "리프컷", image_size="512")
        config = service._client.models.generate_content.call_args.kwargs["config"]
        assert config.image_config.image_size == "512"
        assert config.response_modalities == ["IMAGE"]

    def test_no_image_size_leaves_model_default(self):
        service = self._service()
        service.synthesize_with_reference(self._jpeg(), self._jpeg())
        config = service._client.models.generate_content.call_args.kwargs["config"]
        assert config.image_config is None


# ========== 신고 ==========


class TestResultReport:
    def test_report_is_stored_and_logged(self):
        storage = MagicMock()
        storage.save_report.return_value = "reports/2026-11-01/x.json"
        with patch(
            "api.endpoints.results.get_photo_storage_service", return_value=storage
        ), patch("api.endpoints.results.log_structured") as log, TestClient(
            app
        ) as client:
            response = client.post(
                "/api/results/report",
                json={
                    "result_id": "abcd1234abcd1234",
                    "reason": "face_altered",
                    "comment": "얼굴이 달라요",
                    "device_id": "device-report-01",
                },
            )
        assert response.status_code == 200
        saved = storage.save_report.call_args.args[0]
        assert saved["reason"] == "face_altered"
        assert saved["result_id"] == "abcd1234abcd1234"
        assert "device-report-01" not in str(saved)  # 원본 식별자는 남기지 않는다
        assert saved["comment"] == "얼굴이 달라요"
        # 자유 입력은 로그에 남기지 않는다 (길이만)
        logged = log.call_args.args[1]
        assert "comment" not in logged
        assert logged["comment_length"] == len("얼굴이 달라요")

    def test_unknown_reason_rejected(self):
        with TestClient(app) as client:
            response = client.post(
                "/api/results/report", json={"result_id": "a", "reason": "spam"}
            )
        assert response.status_code == 422
