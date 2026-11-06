"""체형별 아이템 추천 테스트 (분류 순수 함수 + 사진 검증 + 카탈로그 + 엔드포인트)

임계값은 실사진 보정 전 잠정값이다 (models/body_pose_analyzer.py 참고).
이 테스트는 값 자체가 아니라 (1) 세 체형이 모두 도달 가능하고 (2) 가정 분포의
중심이 균형형이며 (3) 사진 검증이 한국어 400 으로 끝나는지를 고정한다.

MediaPipe 모델은 모킹한다 - 엔드포인트 테스트는 관절 좌표만 주입한다.
"""

import io
import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from core.exceptions import FullBodyNotDetectedException, NotFrontalPoseException
from main import app
from models import body_pose_analyzer as bpa
from models.body_pose_analyzer import (
    BROAD_SHOULDER_RATIO,
    LONG_TORSO_LEG_RATIO,
    LOWER_VOLUME_RATIO,
    classify_body_from_ratios,
    compute_body_ratios,
)
from services.body_fashion_service import BodyFashionService, CATALOG_PATH

WIDTH, HEIGHT = 600, 1200


def make_landmarks(
    shoulder_hip=1.50, leg=0.54, shoulder_torso=0.70, visibility=0.95, ankle_y=0.94
):
    """주어진 비율이 나오는 정면 전신 관절 좌표 (세로 600x1200 사진 기준)"""
    nose_y, shoulder_y = 0.08, 0.22
    hip_y = ankle_y - leg * (ankle_y - nose_y)
    shoulder_px = shoulder_torso * (hip_y - shoulder_y) * HEIGHT
    hip_px = shoulder_px / shoulder_hip

    landmarks = [(0.5, 0.5, visibility)] * 33
    landmarks[bpa.NOSE] = (0.5, nose_y, visibility)
    # 정면이면 사람의 왼쪽(11/23/...)이 사진의 오른쪽에 온다
    for left, right, half_px, y in (
        (bpa.LEFT_SHOULDER, bpa.RIGHT_SHOULDER, shoulder_px / 2, shoulder_y),
        (bpa.LEFT_HIP, bpa.RIGHT_HIP, hip_px / 2, hip_y),
        (bpa.LEFT_KNEE, bpa.RIGHT_KNEE, hip_px / 2, (hip_y + ankle_y) / 2),
        (bpa.LEFT_ANKLE, bpa.RIGHT_ANKLE, hip_px / 2, ankle_y),
    ):
        landmarks[left] = (0.5 + half_px / WIDTH, y, visibility)
        landmarks[right] = (0.5 - half_px / WIDTH, y, visibility)
    return landmarks


# ========== 분류 (순수 함수) ==========


class TestClassifyBodyFromRatios:
    @pytest.mark.parametrize("gender, ratio", [("male", 1.50), ("female", 1.38)])
    def test_assumed_population_center_is_balanced(self, gender, ratio):
        body_type, flags = classify_body_from_ratios(ratio, 0.54, gender)
        assert body_type == "balanced"
        assert flags == []

    @pytest.mark.parametrize("gender", ["male", "female"])
    def test_all_body_types_reachable(self, gender):
        broad = BROAD_SHOULDER_RATIO[gender] + 0.01
        lower = LOWER_VOLUME_RATIO[gender] - 0.01
        assert classify_body_from_ratios(broad, 0.54, gender)[0] == "broad_shoulder"
        assert classify_body_from_ratios(lower, 0.54, gender)[0] == "lower_volume"

    def test_thresholds_are_gender_specific(self):
        # 여성 기준으로는 어깨 넓은형, 남성 기준으로는 균형형
        ratio = 1.50
        assert classify_body_from_ratios(ratio, 0.54, "female")[0] == "broad_shoulder"
        assert classify_body_from_ratios(ratio, 0.54, "male")[0] == "balanced"

    def test_threshold_order_leaves_balanced_band(self):
        for gender in ("male", "female"):
            assert LOWER_VOLUME_RATIO[gender] < BROAD_SHOULDER_RATIO[gender]

    def test_boundaries_are_balanced(self):
        assert classify_body_from_ratios(1.57, 0.54, "male")[0] == "balanced"
        assert classify_body_from_ratios(1.43, 0.54, "male")[0] == "balanced"

    def test_short_leg_ratio_flags_long_torso(self):
        _, flags = classify_body_from_ratios(1.50, LONG_TORSO_LEG_RATIO - 0.01, "male")
        assert flags == ["long_torso"]
        _, flags = classify_body_from_ratios(1.50, LONG_TORSO_LEG_RATIO, "male")
        assert flags == []

    def test_unknown_gender_uses_male_thresholds(self):
        assert classify_body_from_ratios(1.50, 0.54, "neutral")[0] == "balanced"


# ========== 비율 계산 + 사진 검증 (순수 함수) ==========


class TestComputeBodyRatios:
    def test_recovers_ratios_from_landmarks(self):
        ratios = compute_body_ratios(
            make_landmarks(shoulder_hip=1.6, leg=0.53, shoulder_torso=0.7),
            WIDTH,
            HEIGHT,
        )
        assert ratios.shoulder_hip_ratio == pytest.approx(1.6, abs=1e-6)
        assert ratios.leg_ratio == pytest.approx(0.53, abs=1e-6)
        assert ratios.shoulder_torso_ratio == pytest.approx(0.7, abs=1e-6)

    def test_uses_pixel_space_not_normalized(self):
        # 같은 정규화 좌표라도 가로가 넓은 사진이면 어깨 폭 비율이 달라져야 한다
        landmarks = make_landmarks(shoulder_torso=0.7)
        narrow = compute_body_ratios(landmarks, WIDTH, HEIGHT)
        wide = compute_body_ratios(landmarks, WIDTH * 2, HEIGHT)
        assert wide.shoulder_torso_ratio == pytest.approx(
            narrow.shoulder_torso_ratio * 2
        )

    def test_to_dict_rounds(self):
        ratios = compute_body_ratios(make_landmarks(), WIDTH, HEIGHT)
        assert set(ratios.to_dict()) == {"shoulder_hip", "leg", "shoulder_torso"}
        assert all(v == round(v, 3) for v in ratios.to_dict().values())

    @pytest.mark.parametrize(
        "idx",
        [
            bpa.NOSE,
            bpa.LEFT_SHOULDER,
            bpa.RIGHT_HIP,
            bpa.LEFT_KNEE,
            bpa.RIGHT_ANKLE,
        ],
    )
    def test_invisible_joint_rejected(self, idx):
        landmarks = make_landmarks()
        x, y, _ = landmarks[idx]
        landmarks[idx] = (x, y, 0.3)
        with pytest.raises(FullBodyNotDetectedException):
            compute_body_ratios(landmarks, WIDTH, HEIGHT)

    def test_feet_cut_off_rejected(self):
        # MediaPipe 는 프레임 밖 발목도 추정한다 - visibility 가 높아도 좌표로 거른다
        landmarks = make_landmarks(ankle_y=1.08)
        with pytest.raises(FullBodyNotDetectedException):
            compute_body_ratios(landmarks, WIDTH, HEIGHT)

    def test_side_view_rejected(self):
        with pytest.raises(NotFrontalPoseException):
            compute_body_ratios(make_landmarks(shoulder_torso=0.3), WIDTH, HEIGHT)

    def test_missing_landmarks_rejected(self):
        with pytest.raises(FullBodyNotDetectedException):
            compute_body_ratios(make_landmarks()[:20], WIDTH, HEIGHT)

    def test_error_messages_are_korean_guidance(self):
        assert "머리부터 발끝까지" in str(FullBodyNotDetectedException())
        assert "정면" in str(NotFrontalPoseException())


# ========== 카탈로그 ==========


def _all_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _all_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _all_strings(v)


class TestBodyFashionService:
    def setup_method(self):
        self.service = BodyFashionService()

    @pytest.mark.parametrize(
        "body_type", ["broad_shoulder", "lower_volume", "balanced"]
    )
    @pytest.mark.parametrize("gender", ["male", "female"])
    @pytest.mark.parametrize("flags", [[], ["long_torso"]])
    def test_every_combination_builds(self, body_type, gender, flags):
        result = self.service.build_recommendation(body_type, flags, gender)
        assert result["body_type"] == body_type
        assert result["body_type_label"]
        assert result["description"]
        assert result["recommended"] and result["avoid"]
        assert "쿠팡 파트너스" in result["disclosure"]
        assert [s["key"] for s in result["secondary"]] == flags

        recommended = {fit["keyword"] for fit in result["recommended"]}
        avoid = {fit["keyword"] for fit in result["avoid"]}
        assert not recommended & avoid, "같은 핏을 추천하면서 피하라고 하면 안 된다"

    def test_long_torso_tips_are_appended_without_duplicates(self):
        base = self.service.build_recommendation("balanced", [], "female")
        merged = self.service.build_recommendation("balanced", ["long_torso"], "female")
        keywords = [fit["keyword"] for fit in merged["recommended"]]
        assert keywords[: len(base["recommended"])] == [
            fit["keyword"] for fit in base["recommended"]
        ]
        assert len(keywords) == len(set(keywords))
        assert "크롭 상의" in keywords

    def test_body_type_labels(self):
        labels = {
            key: self.service.catalog["body_types"][key]["label"]
            for key in ("broad_shoulder", "lower_volume", "balanced")
        }
        assert labels == {
            "broad_shoulder": "어깨 넓은형",
            "lower_volume": "하체 볼륨형",
            "balanced": "균형형",
        }
        assert self.service.catalog["proportions"]["long_torso"]["label"] == (
            "상체 긴 비율"
        )

    def test_inactive_links_are_hidden(self, tmp_path):
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        for link in catalog["links"]:
            link["active"] = False
            link["affiliate_url"] = "PLACEHOLDER"
        path = tmp_path / "body_catalog.json"
        path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")

        service = BodyFashionService(path)
        assert service.get_product("fashion-m-01") is None
        result = service.build_recommendation("balanced", [], "male")
        assert all(fit["product_id"] is None for fit in result["recommended"])
        assert "items" not in result

    def test_live_catalog_links_are_coupang_partner_links(self):
        for link in self.service.catalog["links"]:
            if link["active"]:
                assert link["affiliate_url"].startswith("https://link.coupang.com/")
            else:
                assert link["affiliate_url"] == "PLACEHOLDER"

    def test_active_links_attach_to_keywords_without_url(self, tmp_path):
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        for link in catalog["links"]:
            link["active"] = True
            link["affiliate_url"] = "https://link.coupang.com/a/test"
        path = tmp_path / "body_catalog.json"
        path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")

        result = BodyFashionService(path).build_recommendation(
            "broad_shoulder", ["long_torso"], "male"
        )
        by_keyword = {
            fit["keyword"]: fit["product_id"] for fit in result["recommended"]
        }
        assert by_keyword["와이드 슬랙스"] == "fashion-m-01"
        assert by_keyword["하이웨이스트 팬츠"] == "fashion-m-13"
        # 스타일링 팁은 살 물건이 아니라 링크가 없다
        assert by_keyword["상의 앞 넣어 입기"] is None
        assert all("affiliate_url" not in fit for fit in result["recommended"])

    def test_links_follow_gender(self, tmp_path):
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        for link in catalog["links"]:
            link["active"] = True
            link["affiliate_url"] = "https://link.coupang.com/a/test"
        path = tmp_path / "body_catalog.json"
        path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")

        result = BodyFashionService(path).build_recommendation(
            "broad_shoulder", [], "female"
        )
        ids = [fit["product_id"] for fit in result["recommended"]]
        assert ids and all(pid.startswith("fashion-f-") for pid in ids)

    def test_every_recommendation_is_linked_or_a_tip(self):
        catalog = self.service.catalog
        links = {(link["gender"], link["keyword"]) for link in catalog["links"]}
        assert all(
            link["product_id"].startswith("fashion-") for link in catalog["links"]
        )
        tips = {("male", "상의 앞 넣어 입기"), ("female", "상의 넣어 입기")}
        sections = list(catalog["body_types"].values()) + list(
            catalog["proportions"].values()
        )
        for section in sections:
            for gender in ("male", "female"):
                for fit in section[gender]["recommended"]:
                    assert (gender, fit["keyword"]) in links | tips, fit["keyword"]

    def test_no_weight_or_fat_wording(self):
        banned = ("뚱뚱", "날씬", "체중", "체지방", "다이어트", "살쪄", "살이", "비만")
        user_facing = {k: v for k, v in self.service.catalog.items() if k != "_readme"}
        for text in _all_strings(user_facing):
            assert not any(word in text for word in banned), text


# ========== 엔드포인트 ==========


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 128), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _no_rate_limit():
    from api.endpoints import fashion as fashion_module

    previous = fashion_module.limiter.enabled
    fashion_module.limiter.enabled = False
    yield
    fashion_module.limiter.enabled = previous


@pytest.fixture
def pose_result():
    """모델 대신 관절 좌표를 돌려주는 분석기 (결과는 테스트에서 바꾼다)"""
    analyzer = MagicMock()
    analyzer.extract_landmarks.return_value = (make_landmarks(), WIDTH, HEIGHT)
    with patch("api.endpoints.fashion.get_body_pose_analyzer", return_value=analyzer):
        yield analyzer


def _post(client, gender="male", filename="body.png", content=None):
    return client.post(
        "/api/fashion/body-analyze",
        files={"file": (filename, content or _png_bytes(), "image/png")},
        data={"gender": gender},
    )


class TestBodyAnalyzeEndpoint:
    def test_success_response_shape(self, client, pose_result):
        pose_result.extract_landmarks.return_value = (
            make_landmarks(shoulder_hip=1.65, leg=0.50),
            WIDTH,
            HEIGHT,
        )
        response = _post(client, gender="female")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["body_type"] == "broad_shoulder"
        assert data["body_type_label"] == "어깨 넓은형"
        assert data["description"]
        assert data["gender"] == "female"
        assert [s["key"] for s in data["secondary"]] == ["long_torso"]
        assert data["ratios"] == {
            "shoulder_hip": 1.65,
            "leg": 0.5,
            "shoulder_torso": 0.7,
        }
        assert data["recommended"][0]["keyword"] == "브이넥 블라우스"
        assert data["avoid"]
        # 링크 ID만 나가고 URL 은 클릭 API 에서만 준다
        assert data["recommended"][0]["product_id"] == "fashion-f-01"
        assert "affiliate_url" not in data["recommended"][0]
        assert "쿠팡 파트너스" in data["disclosure"]

    def test_logs_ratios_without_image(self, client, pose_result):
        with patch("api.endpoints.fashion.log_structured") as log:
            response = _post(client)
        assert response.status_code == 200
        event, payload = log.call_args.args
        assert event == "body_analysis_complete"
        assert payload["body_type"] == "balanced"
        assert payload["gender"] == "male"
        assert payload["shoulder_hip_ratio"] == pytest.approx(1.5)
        assert "processing_time" in payload
        assert payload["authenticated"] is False
        assert not any(isinstance(v, (bytes, bytearray)) for v in payload.values())

    def test_no_person_returns_korean_400(self, client, pose_result):
        pose_result.extract_landmarks.return_value = None
        with patch("api.endpoints.fashion.log_structured") as log:
            response = _post(client)

        assert response.status_code == 400
        body = response.json()
        assert body["success"] is False
        assert body["error"] == "fullbodynotdetected"
        assert "머리부터 발끝까지" in body["message"]
        event, payload = log.call_args.args
        assert event == "body_analysis_error"
        assert payload["error_type"] == "full_body_not_detected"

    def test_partial_body_returns_400(self, client, pose_result):
        pose_result.extract_landmarks.return_value = (
            make_landmarks(ankle_y=1.1),
            WIDTH,
            HEIGHT,
        )
        response = _post(client)
        assert response.status_code == 400
        assert response.json()["error"] == "fullbodynotdetected"

    def test_side_view_returns_400(self, client, pose_result):
        pose_result.extract_landmarks.return_value = (
            make_landmarks(shoulder_torso=0.3),
            WIDTH,
            HEIGHT,
        )
        response = _post(client)
        assert response.status_code == 400
        body = response.json()
        assert body["error"] == "notfrontalpose"
        assert "정면" in body["message"]

    def test_invalid_file_returns_400(self, client, pose_result):
        response = _post(client, filename="body.png", content=b"not an image")
        assert response.status_code == 400
        assert response.json()["error"] == "invalidfileformat"
        pose_result.extract_landmarks.assert_not_called()

    def test_invalid_gender_rejected(self, client, pose_result):
        response = _post(client, gender="neutral")
        assert response.status_code == 422
        pose_result.extract_landmarks.assert_not_called()

    def test_model_failure_returns_500(self, client, pose_result):
        pose_result.extract_landmarks.side_effect = RuntimeError("boom")
        response = _post(client)
        assert response.status_code == 500


# ========== 클릭 (POST /products/click 확장) ==========


@pytest.fixture
def mock_click_service():
    service = MagicMock()
    service.log_click.return_value = True
    with patch(
        "api.endpoints.products.get_affiliate_click_service", return_value=service
    ):
        yield service


class TestFashionItemClick:
    def test_inactive_item_is_404(self, client, mock_click_service):
        fashion = MagicMock()
        fashion.get_product.return_value = None
        with patch(
            "api.endpoints.products.get_body_fashion_service", return_value=fashion
        ):
            response = client.post(
                "/api/products/click",
                json={"product_id": "fashion-m-01", "source": "body_analysis"},
            )
        assert response.status_code == 404
        mock_click_service.log_click.assert_not_called()

    def test_active_item_returns_link_and_logs(self, client, mock_click_service):
        product = {
            "product_id": "fashion-m-01",
            "category": "pants",
            "affiliate_url": "https://link.coupang.com/a/test",
            "active": True,
        }
        fashion = MagicMock()
        fashion.get_product.return_value = product
        fashion.disclosure = "쿠팡 파트너스 문구"
        with patch(
            "api.endpoints.products.get_body_fashion_service", return_value=fashion
        ):
            response = client.post(
                "/api/products/click",
                json={
                    "product_id": "fashion-m-01",
                    "source": "body_analysis",
                    "hair_profile": {"body_type": "broad_shoulder", "gender": "male"},
                    "device_id": "a1b2c3d4e5f60718",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["affiliate_url"] == "https://link.coupang.com/a/test"
        assert data["disclosure"] == "쿠팡 파트너스 문구"
        args = mock_click_service.log_click.call_args.args
        assert args[0] == "device#a1b2c3d4e5f60718"
        assert args[1] is product
        assert args[3] == "body_analysis"
        assert args[4] == {"body_type": "broad_shoulder", "gender": "male"}


# ========== 사진 저장 (동의한 요청만, 헤어 사진과 분리) ==========


@pytest.fixture
def storage():
    service = MagicMock()
    service.save_body_photo.return_value = "body/originals/x/1.png"
    with patch("api.endpoints.fashion.get_photo_storage_service", return_value=service):
        yield service


def _post_with(client, **data):
    return client.post(
        "/api/fashion/body-analyze",
        files={"file": ("body.png", _png_bytes(), "image/png")},
        data={"gender": "male", **data},
    )


class TestBodyPhotoSaving:
    def test_no_consent_never_saves(self, client, pose_result, storage):
        response = _post_with(client, device_id="a1b2c3d4e5f60718")
        assert response.status_code == 200
        storage.save_body_photo.assert_not_called()

    def test_consent_saves_under_device_with_ratios(self, client, pose_result, storage):
        with patch("api.endpoints.fashion.log_structured") as log:
            response = _post_with(
                client, save_consent="true", device_id="a1b2c3d4e5f60718"
            )
        assert response.status_code == 200
        subject, image_bytes, metadata = storage.save_body_photo.call_args.args
        assert subject == "device-a1b2c3d4e5f60718"
        assert image_bytes.startswith(b"\x89PNG")
        assert metadata["body-type"] == "balanced"
        assert metadata["gender"] == "male"
        assert metadata["shoulder-hip"] == "1.500"
        assert all(isinstance(v, str) and v.isascii() for v in metadata.values())
        _, payload = log.call_args.args
        assert payload["save_consent"] is True and payload["photo_saved"] is True

    def test_logged_in_user_is_subject(self, client, pose_result, storage):
        from main import app as main_app
        from core.jwt_auth import get_optional_user_id

        main_app.dependency_overrides[get_optional_user_id] = lambda: "u-123"
        try:
            _post_with(client, save_consent="true", device_id="a1b2c3d4e5f60718")
        finally:
            main_app.dependency_overrides.pop(get_optional_user_id, None)
        assert storage.save_body_photo.call_args.args[0] == "user-u-123"

    @pytest.mark.parametrize("device_id", [None, "../../etc", "short"])
    def test_unsafe_or_missing_device_is_anonymous(
        self, client, pose_result, storage, device_id
    ):
        data = {"save_consent": "true"}
        if device_id:
            data["device_id"] = device_id
        _post_with(client, **data)
        assert storage.save_body_photo.call_args.args[0] == "anonymous"

    def test_pose_failure_photo_is_saved_when_consented(
        self, client, pose_result, storage
    ):
        pose_result.extract_landmarks.return_value = None
        response = _post_with(client, save_consent="true", device_id="a1b2c3d4e5f60718")
        assert response.status_code == 400
        metadata = storage.save_body_photo.call_args.args[2]
        assert metadata["error"] == "full_body_not_detected"

    def test_invalid_file_is_not_saved(self, client, pose_result, storage):
        response = client.post(
            "/api/fashion/body-analyze",
            files={"file": ("body.png", b"not an image", "image/png")},
            data={"save_consent": "true"},
        )
        assert response.status_code == 400
        storage.save_body_photo.assert_not_called()

    def test_storage_failure_does_not_break_analysis(
        self, client, pose_result, storage
    ):
        storage.save_body_photo.side_effect = RuntimeError("s3 down")
        response = _post_with(client, save_consent="true")
        assert response.status_code == 200
        assert response.json()["success"] is True


class TestBodyPhotoStorageService:
    def test_key_is_separate_from_hair_originals(self, monkeypatch):
        from services.photo_storage_service import PhotoStorageService

        monkeypatch.setenv("PHOTO_S3_BUCKET", "test-bucket")
        service = PhotoStorageService()
        service._client = MagicMock()

        key = service.save_body_photo(
            "device-abc12345", _png_bytes(), {"gender": "male"}
        )

        assert key.startswith("body/originals/device-abc12345/")
        assert key.endswith(".png")
        kwargs = service._client.put_object.call_args.kwargs
        assert kwargs["Bucket"] == "test-bucket"
        assert kwargs["ContentType"] == "image/png"
        assert kwargs["Metadata"]["body-photo-consent"] == "true"
        assert not kwargs["Key"].startswith("originals/")

    def test_disabled_without_bucket(self, monkeypatch):
        from services.photo_storage_service import PhotoStorageService

        monkeypatch.setenv("PHOTO_S3_BUCKET", "")
        assert PhotoStorageService().save_body_photo("x", b"1", {}) is None


def test_lifecycle_merge_keeps_existing_rules():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "apply_body_photo_lifecycle.py"
    )
    spec = importlib.util.spec_from_file_location("apply_body_photo_lifecycle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    existing = [
        {"ID": "expire-cache", "Filter": {"Prefix": "cache/"}},
        {"ID": "expire-body-originals", "Expiration": {"Days": 30}},
    ]
    rules = module.merged_rules(existing)
    by_id = {r["ID"]: r for r in rules}
    assert set(by_id) == {"expire-cache", "expire-body-originals"}
    assert by_id["expire-body-originals"]["Expiration"]["Days"] == 365
    assert by_id["expire-body-originals"]["Filter"]["Prefix"] == "body/originals/"
