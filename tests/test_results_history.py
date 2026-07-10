"""결과 히스토리 API 테스트 (S3 모킹)"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from core.jwt_auth import create_access_token
from main import app
from services.photo_storage_service import PhotoStorageService

USER_ID = "history-user-id"
PREFIX = f"results/{USER_ID}/"


def _s3_object(key: str):
    return {
        "Key": key,
        "LastModified": datetime(2026, 7, 1, tzinfo=timezone.utc),
    }


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv("PHOTO_S3_BUCKET", "test-bucket")
    svc = PhotoStorageService()
    svc._client = MagicMock()
    svc._client.generate_presigned_url.return_value = "https://s3.example/presigned"
    svc._client.head_object.return_value = {"Metadata": {}}
    return svc


class TestListUserResults:
    def test_newest_first(self, service):
        """S3는 오름차순으로 반환하지만 응답은 최신순"""
        service._client.list_objects_v2.return_value = {
            "Contents": [
                _s3_object(f"{PREFIX}20260701T010000_aaaa.png"),
                _s3_object(f"{PREFIX}20260702T010000_bbbb.png"),
                _s3_object(f"{PREFIX}20260703T010000_cccc.png"),
            ],
            "IsTruncated": False,
        }

        page = service.list_user_results(USER_ID)

        keys = [item["key"] for item in page["items"]]
        assert keys == [
            f"{PREFIX}20260703T010000_cccc.png",
            f"{PREFIX}20260702T010000_bbbb.png",
            f"{PREFIX}20260701T010000_aaaa.png",
        ]
        assert page["next_token"] is None
        # 회원 prefix로만 조회하는지 확인
        list_kwargs = service._client.list_objects_v2.call_args.kwargs
        assert list_kwargs["Prefix"] == PREFIX

    def test_pagination(self, service):
        service._client.list_objects_v2.return_value = {
            "Contents": [
                _s3_object(f"{PREFIX}20260701T010000_aaaa.png"),
                _s3_object(f"{PREFIX}20260702T010000_bbbb.png"),
                _s3_object(f"{PREFIX}20260703T010000_cccc.png"),
            ],
            "IsTruncated": False,
        }

        first = service.list_user_results(USER_ID, limit=2)
        assert len(first["items"]) == 2
        assert first["next_token"] == f"{PREFIX}20260702T010000_bbbb.png"

        second = service.list_user_results(
            USER_ID, limit=2, continuation_token=first["next_token"]
        )
        assert [item["key"] for item in second["items"]] == [
            f"{PREFIX}20260701T010000_aaaa.png"
        ]
        assert second["next_token"] is None

    def test_truncated_s3_listing_is_followed(self, service):
        """1000개 초과 시 S3 ContinuationToken으로 나머지도 수집"""
        service._client.list_objects_v2.side_effect = [
            {
                "Contents": [_s3_object(f"{PREFIX}20260701T010000_aaaa.png")],
                "IsTruncated": True,
                "NextContinuationToken": "s3-token",
            },
            {
                "Contents": [_s3_object(f"{PREFIX}20260702T010000_bbbb.png")],
                "IsTruncated": False,
            },
        ]

        page = service.list_user_results(USER_ID)

        assert len(page["items"]) == 2
        second_call = service._client.list_objects_v2.call_args_list[1].kwargs
        assert second_call["ContinuationToken"] == "s3-token"

    def test_hairstyle_metadata_decoded(self, service):
        """저장 시 URL 인코딩된 한글 스타일명 복원"""
        service._client.list_objects_v2.return_value = {
            "Contents": [_s3_object(f"{PREFIX}20260701T010000_aaaa.png")],
            "IsTruncated": False,
        }
        service._client.head_object.return_value = {
            "Metadata": {"hairstyle": quote("레이어드컷")}
        }

        page = service.list_user_results(USER_ID)

        assert page["items"][0]["hairstyle"] == "레이어드컷"
        assert page["items"][0]["url"] == "https://s3.example/presigned"
        assert page["items"][0]["created_at"] == "2026-07-01T00:00:00+00:00"

    def test_head_failure_does_not_break_listing(self, service):
        service._client.list_objects_v2.return_value = {
            "Contents": [_s3_object(f"{PREFIX}20260701T010000_aaaa.png")],
            "IsTruncated": False,
        }
        service._client.head_object.side_effect = Exception("S3 down")

        page = service.list_user_results(USER_ID)

        assert len(page["items"]) == 1
        assert page["items"][0]["hairstyle"] is None

    def test_disabled_storage_returns_empty(self, monkeypatch):
        monkeypatch.delenv("PHOTO_S3_BUCKET", raising=False)
        svc = PhotoStorageService()

        with patch("services.photo_storage_service.settings") as mock_settings:
            mock_settings.PHOTO_S3_BUCKET = ""
            page = svc.list_user_results(USER_ID)

        assert page == {"items": [], "next_token": None}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers():
    token = create_access_token(USER_ID)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def mock_storage():
    service = MagicMock()
    service.list_user_results.return_value = {
        "items": [
            {
                "key": f"{PREFIX}20260701T010000_aaaa.png",
                "url": "https://s3.example/presigned",
                "hairstyle": "레이어드컷",
                "created_at": "2026-07-01T00:00:00+00:00",
            }
        ],
        "next_token": None,
    }
    with patch("api.endpoints.results.get_photo_storage_service", return_value=service):
        yield service


class TestMyResultsEndpoint:
    def test_requires_auth(self, client, mock_storage):
        response = client.get("/api/me/results")
        assert response.status_code == 401

    def test_returns_results(self, client, auth_headers, mock_storage):
        response = client.get("/api/me/results", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["next_token"] is None
        assert data["results"][0]["hairstyle"] == "레이어드컷"
        # JWT의 user_id로만 조회 (다른 회원 결과 접근 불가)
        assert mock_storage.list_user_results.call_args.args[0] == USER_ID

    def test_pagination_params_passed(self, client, auth_headers, mock_storage):
        response = client.get(
            "/api/me/results?limit=5&continuation_token=some-key",
            headers=auth_headers,
        )

        assert response.status_code == 200
        args = mock_storage.list_user_results.call_args.args
        assert args[1] == 5
        assert args[2] == "some-key"

    def test_invalid_limit_rejected(self, client, auth_headers, mock_storage):
        assert (
            client.get("/api/me/results?limit=0", headers=auth_headers).status_code
            == 422
        )
        assert (
            client.get("/api/me/results?limit=999", headers=auth_headers).status_code
            == 422
        )

    def test_s3_error_hides_internals(self, client, auth_headers, mock_storage):
        mock_storage.list_user_results.side_effect = Exception("NoSuchBucket: secret")

        response = client.get("/api/me/results", headers=auth_headers)

        assert response.status_code == 500
        assert "NoSuchBucket" not in response.text
