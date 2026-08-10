# -*- coding: utf-8 -*-
"""스타일 예시 이미지 서비스/엔드포인트 테스트"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import services.style_image_service as sis
import api.endpoints.style_images as style_images_ep


# ========== 픽스처 ==========


@pytest.fixture
def service(tmp_path, monkeypatch):
    """임시 styles.json/image_manifest.json으로 서비스 생성"""
    styles = {
        "ml_styles": [
            {
                "hairstyle_id": 72,
                "raw_name": "크롭 컷 (Crop Cut)",
                "canonical": "크롭컷",
                "image_key": "크롭컷",
                "gender": "male",
            },
            {
                "hairstyle_id": 62,
                "raw_name": "울프 컷",
                "canonical": "울프컷",
                "image_key": "울프컷",
                "gender": "unisex",
            },
        ],
        "trending_styles": [
            {
                "style_name": "보브 컷",
                "canonical": "보브컷",
                "image_key": "보브컷",
                "gender": "female",
            }
        ],
    }
    manifest = {
        "male/크롭컷": {"file": "male_072_크롭컷_1.webp", "alts": []},
        "male/울프컷": {"file": "male_062_울프컷_1.webp", "alts": []},
        "female/울프컷": {"file": "female_062_울프컷_1.webp", "alts": []},
        "female/보브컷": {"file": "female_031_보브컷_1.webp", "alts": []},
    }
    styles_path = tmp_path / "styles.json"
    manifest_path = tmp_path / "image_manifest.json"
    styles_path.write_text(json.dumps(styles, ensure_ascii=False), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(sis, "_STYLES_PATH", styles_path)
    monkeypatch.setattr(sis, "_MANIFEST_PATH", manifest_path)
    return sis.StyleImageService()


# ========== StyleImageService ==========


class TestStyleImageService:
    def test_resolve_by_hairstyle_id(self, service):
        assert (
            service.resolve_filename(hairstyle_id=72, gender="male")
            == "male_072_크롭컷_1.webp"
        )

    def test_resolve_by_raw_name(self, service):
        assert (
            service.resolve_filename(style_name="크롭 컷 (Crop Cut)", gender="male")
            == "male_072_크롭컷_1.webp"
        )

    def test_resolve_by_normalized_name(self, service):
        # 띄어쓰기 변형도 매칭
        assert (
            service.resolve_filename(style_name="크롭컷", gender="male")
            == "male_072_크롭컷_1.webp"
        )

    def test_resolve_trending_by_name(self, service):
        assert (
            service.resolve_filename(style_name="보브 컷", gender="female")
            == "female_031_보브컷_1.webp"
        )

    def test_unisex_follows_requested_gender(self, service):
        assert (
            service.resolve_filename(style_name="울프컷", gender="female")
            == "female_062_울프컷_1.webp"
        )
        assert (
            service.resolve_filename(style_name="울프컷", gender="male")
            == "male_062_울프컷_1.webp"
        )

    def test_gender_fallback_to_other_gender(self, service):
        # 남성 전용 스타일이 여성 사용자에게 추천된 경우 남성 이미지로 폴백
        assert (
            service.resolve_filename(style_name="크롭컷", gender="female")
            == "male_072_크롭컷_1.webp"
        )

    def test_unknown_style_returns_none(self, service):
        assert service.resolve_filename(style_name="없는스타일", gender="male") is None

    def test_build_image_url_absolute(self, service):
        url = service.build_image_url(
            hairstyle_id=72, gender="male", base_url="https://api.example.com/"
        )
        assert url.startswith("https://api.example.com/api/styles/images/")
        assert url.endswith(".webp")
        assert " " not in url  # 파일명은 URL 인코딩됨

    def test_build_image_url_unknown_returns_none(self, service):
        assert (
            service.build_image_url(style_name="없는스타일", base_url="https://x/")
            is None
        )

    def test_missing_mapping_files_disable_service(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sis, "_STYLES_PATH", tmp_path / "no.json")
        monkeypatch.setattr(sis, "_MANIFEST_PATH", tmp_path / "no2.json")
        svc = sis.StyleImageService()
        assert svc.enabled is False
        assert svc.resolve_filename(style_name="크롭컷", gender="male") is None


# ========== 서빙 엔드포인트 ==========


@pytest.fixture
def image_client(tmp_path, monkeypatch):
    webp_dir = tmp_path / "webp"
    webp_dir.mkdir()
    (webp_dir / "male_072_크롭컷_1.webp").write_bytes(b"RIFF....WEBPfake")
    monkeypatch.setattr(style_images_ep, "_WEBP_DIR", webp_dir)

    app = FastAPI()
    app.include_router(style_images_ep.router, prefix="/api")
    return TestClient(app)


class TestStyleImageEndpoint:
    def test_serves_existing_image_with_cache_headers(self, image_client):
        res = image_client.get("/api/styles/images/male_072_크롭컷_1.webp")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/webp"
        assert "max-age=31536000" in res.headers["cache-control"]

    def test_missing_image_returns_404(self, image_client):
        res = image_client.get("/api/styles/images/none.webp")
        assert res.status_code == 404

    def test_rejects_non_webp(self, image_client):
        res = image_client.get("/api/styles/images/evil.py")
        assert res.status_code == 404

    def test_rejects_path_traversal(self, image_client):
        res = image_client.get("/api/styles/images/..%2F..%2Fconfig%2Fsettings.py")
        assert res.status_code in (404, 400)
