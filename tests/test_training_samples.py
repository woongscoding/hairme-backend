"""학습 데이터(원본 + 합성 결과) 저장 테스트 (S3 모킹)"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

import base64
import json
from unittest.mock import MagicMock, patch

import pytest

from api.endpoints import synthesis
from services.photo_storage_service import PhotoStorageService

JPEG_BYTES = b"\xff\xd8\xff\xe0fake-jpeg"
RESULT_B64 = base64.b64encode(b"\x89PNGfake-result").decode()
DEVICE_ID = "device-abcdef1234567890"


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv("PHOTO_S3_BUCKET", "test-bucket")
    svc = PhotoStorageService()
    svc._client = MagicMock()
    svc._client.generate_presigned_url.return_value = "https://s3.example/presigned"
    return svc


def _put_keys(service):
    return [c.kwargs["Key"] for c in service._client.put_object.call_args_list]


def _training_keys(service):
    return [k for k in _put_keys(service) if k.startswith("training/")]


def _store(service, user=None, **kwargs):
    repo = MagicMock()
    repo.get_by_id.return_value = user
    with patch.object(
        synthesis, "get_photo_storage_service", return_value=service
    ), patch.object(synthesis, "get_user_repository", return_value=repo):
        return synthesis._store_result(
            kwargs.pop("user_id", None),
            JPEG_BYTES,
            "cachekey",
            RESULT_B64,
            "png",
            "투블럭컷",
            **kwargs,
        )


class TestSaveTrainingSample:
    def test_saves_original_result_and_meta_together(self, service):
        prefix = service.save_training_sample(
            "user-u1", JPEG_BYTES, RESULT_B64, "png", {"gender": "male"}
        )

        assert prefix.startswith("training/user-u1/")
        keys = _put_keys(service)
        assert keys == [
            f"{prefix}original.jpg",
            f"{prefix}result.png",
            f"{prefix}meta.json",
        ]
        meta_call = service._client.put_object.call_args_list[2]
        meta = json.loads(meta_call.kwargs["Body"].decode("utf-8"))
        assert meta["gender"] == "male"
        assert "created_at" in meta

    def test_s3_failure_returns_none(self, service):
        service._client.put_object.side_effect = Exception("boom")
        assert (
            service.save_training_sample("user-u1", JPEG_BYTES, RESULT_B64, "png", {})
            is None
        )


class TestStoreResultTrainingConsent:
    def test_anonymous_with_consent_is_saved(self, service):
        _store(service, device_id=DEVICE_ID, device_consent=True)

        keys = _training_keys(service)
        assert len(keys) == 3
        assert all(k.startswith(f"training/device-{DEVICE_ID}/") for k in keys)

    def test_anonymous_without_consent_is_not_saved(self, service):
        _store(service, device_id=DEVICE_ID, device_consent=False)
        assert _training_keys(service) == []

    def test_anonymous_invalid_device_id_is_not_saved(self, service):
        _store(service, device_id="x", device_consent=True)
        assert _training_keys(service) == []

    def test_member_uses_account_consent(self, service):
        _store(service, user={"training_consent": True}, user_id="u1")

        keys = _training_keys(service)
        assert len(keys) == 3
        assert all(k.startswith("training/user-u1/") for k in keys)

    def test_member_form_consent_does_not_override_account(self, service):
        """회원은 폼 값이 아니라 계정 설정이 기준"""
        _store(
            service,
            user={"training_consent": False},
            user_id="u1",
            device_consent=True,
        )
        assert _training_keys(service) == []
