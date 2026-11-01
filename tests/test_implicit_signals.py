# -*- coding: utf-8 -*-
"""암묵 피드백 (합성·저장·공유 -> 학습 샘플) 테스트

- 라벨/가중치 (base x clip(1/propensity, 1, 5), propensity 없으면 1.0)
- npz 레이아웃이 save_feedback 과 같고 sample_weight / 암묵 메타가 더해진다
- 같은 분석x스타일은 더 강한 신호일 때만 덮어쓴다 (share > save > synthesize > shown_not_chosen)
- shown_not_chosen 은 분석의 첫 synthesize 에서만, 다른 ML 카드에만 (트렌드 제외)
- 새 샘플만 metadata.json 의 pending_count / total 을 올린다
- MLOPS 꺼짐 / 분석 없음 / 측정값 없음 / 카드 없음은 조용히 생략
- POST /api/events/style-action 422/200, 합성 선택 필드(구버전 요청 동일 동작)

S3 는 moto(@pytest.mark.moto), 엔드포인트는 mock. 운영 버킷/테이블에 쓰지 않는다.
"""

import io
import json
import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

import services.mlops.s3_feedback_store as sfs
from services.mlops import implicit_signals as isig
from services.mlops.implicit_signals import (
    SIGNAL_SAVE,
    SIGNAL_SHARE,
    SIGNAL_SHOWN_NOT_CHOSEN,
    SIGNAL_SYNTHESIZE,
    implicit_key,
    record_style_signal,
    sample_weight,
)

BUCKET = "hairme-mlops-unittest"
REGION = "ap-northeast-2"
ANALYSIS_ID = "11111111-2222-3333-4444-555555555555"
EMBED_DIM = 384
N_STYLES = 12


def _analysis(**overrides):
    item = {
        "analysis_id": ANALYSIS_ID,
        "face_shape": "계란형",
        "personal_color": "봄웜",
        "model_version": "v6_feedback_20260909",
        "user_id": "user-should-not-leak",
        "mediapipe_face_ratio": 1.4,
        "mediapipe_forehead_width": 120.0,
        "mediapipe_cheekbone_width": 130.0,
        "mediapipe_jaw_width": 110.0,
        "mediapipe_forehead_ratio": 0.92,
        "mediapipe_jaw_ratio": 0.85,
        "mediapipe_ITA_value": 45.0,
        "mediapipe_hue_value": 15.0,
        # get_analysis 는 Decimal 을 float 로 돌려준다 (정수도 3.0)
        "recommended_styles": [
            {
                "hairstyle_id": 3.0,
                "style_name": "레이어드 컷",
                "source": "ml",
                "rank": 1.0,
                "ml_rank": 2.0,
                "propensity": 0.25,
            },
            {
                "hairstyle_id": 5.0,
                "style_name": "시스루 뱅",
                "source": "ml",
                "rank": 2.0,
                "ml_rank": 1.0,
                "propensity": 0.8,
            },
            # 탐색 꺼짐 시절 카드: ml_rank / propensity 없음
            {"hairstyle_id": 7.0, "style_name": "웨이브 펌", "source": "ml", "rank": 3.0},
            {"hairstyle_id": None, "style_name": "트렌드A", "source": "trending", "rank": 4.0},
            {"hairstyle_id": None, "style_name": "트렌드B", "source": "trending", "rank": 5.0},
        ],
    }
    item.update(overrides)
    return item


# ========== 가중치 (순수 함수) ==========


class TestSampleWeight:
    def test_ips_weighting(self):
        assert sample_weight(SIGNAL_SYNTHESIZE, 0.25) == pytest.approx(0.7 * 4)
        assert sample_weight(SIGNAL_SHOWN_NOT_CHOSEN, 0.8) == pytest.approx(0.3 * 1.25)

    def test_clipped_to_5(self):
        assert sample_weight(SIGNAL_SHARE, 0.1) == pytest.approx(5.0)
        assert sample_weight(SIGNAL_SAVE, 0.01) == pytest.approx(5.0)

    def test_missing_or_invalid_propensity_is_1(self):
        assert sample_weight(SIGNAL_SAVE, None) == pytest.approx(1.0)
        assert sample_weight(SIGNAL_SYNTHESIZE, 0) == pytest.approx(0.7)
        assert sample_weight(SIGNAL_SYNTHESIZE, -0.5) == pytest.approx(0.7)
        assert sample_weight(SIGNAL_SYNTHESIZE, "nan") == pytest.approx(0.7)
        # 1 보다 크면 clip 하한 1
        assert sample_weight(SIGNAL_SHOWN_NOT_CHOSEN, 1.7) == pytest.approx(0.3)

    def test_labels(self):
        assert {k: v[0] for k, v in isig.SIGNAL_SPECS.items()} == {
            "share": 85.0,
            "save": 80.0,
            "synthesize": 55.0,
            "shown_not_chosen": 40.0,
        }

    def test_key_format(self):
        assert (
            implicit_key(ANALYSIS_ID, 42)
            == f"feedback/pending/implicit/{ANALYSIS_ID}_42.npz"
        )

    def test_parsers_ignore_bad_values(self):
        assert isig.parse_rec_rank("2") == 2
        assert isig.parse_rec_rank("0") is None
        assert isig.parse_rec_rank("6") is None
        assert isig.parse_rec_rank("abc") is None
        assert isig.parse_rec_rank("2.5") is None
        assert isig.parse_hairstyle_id("42") == 42
        assert isig.parse_hairstyle_id("-1") is None
        assert isig.parse_hairstyle_id("") is None
        assert isig.parse_analysis_id(ANALYSIS_ID) == ANALYSIS_ID
        assert isig.parse_analysis_id("../../etc") is None
        assert isig.parse_analysis_id("a" * 65) is None
        assert isig.parse_analysis_id(None) is None


# ========== S3 (moto) ==========


@pytest.fixture
def mlops_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MLOPS_ENABLED", "true")
    embeddings = np.random.default_rng(0).random((N_STYLES, EMBED_DIM)).astype(
        np.float32
    )
    path = tmp_path / "style_embeddings.npz"
    np.savez(path, styles=np.array([f"s{i}" for i in range(N_STYLES)]), embeddings=embeddings)
    monkeypatch.setattr(sfs, "LOCAL_EMBEDDINGS_PATH", str(path))
    return embeddings


@pytest.fixture
def s3_store(mlops_env):
    import boto3
    from moto import mock_aws

    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION}
        )
        store = sfs.S3FeedbackStore(bucket_name=BUCKET, region=REGION)
        assert store.enabled
        yield store


def _record(store, signal, analysis=None, **kwargs):
    item = _analysis() if analysis is None else analysis
    return record_style_signal(
        ANALYSIS_ID,
        signal,
        store=store,
        analysis_getter=lambda _id: item,
        **kwargs,
    )


def _implicit_keys(store):
    response = store.s3_client.list_objects_v2(
        Bucket=BUCKET, Prefix="feedback/pending/implicit/"
    )
    return sorted(obj["Key"] for obj in response.get("Contents", []))


def _load(store, hairstyle_id):
    body = store.s3_client.get_object(
        Bucket=BUCKET, Key=implicit_key(ANALYSIS_ID, hairstyle_id)
    )["Body"].read()
    data = np.load(io.BytesIO(body), allow_pickle=False)
    return data, json.loads(str(data["metadata"][0]))


def _head_signal(store, hairstyle_id):
    return store.s3_client.head_object(
        Bucket=BUCKET, Key=implicit_key(ANALYSIS_ID, hairstyle_id)
    )["Metadata"].get("signal-type")


def _counts(store):
    metadata = store._get_metadata()
    return metadata.get("pending_count", 0), metadata.get("total_feedback_count", 0)


@pytest.mark.moto
class TestRecording:
    def test_first_synthesize_creates_chosen_and_shown_not_chosen(self, s3_store, mlops_env):
        result = _record(s3_store, SIGNAL_SYNTHESIZE, hairstyle_id=3, rec_rank=1)

        assert result["recorded"] is True
        assert result["reason"] == "created"
        assert result["shown_not_chosen"] == 2
        # ML 카드 3장만 - 트렌드 카드는 샘플 없음
        assert _implicit_keys(s3_store) == sorted(
            implicit_key(ANALYSIS_ID, hid) for hid in (3, 5, 7)
        )

        data, meta = _load(s3_store, 3)
        assert data["ground_truth"].tolist() == [55.0]
        assert data["sample_weight"].dtype == np.float32
        assert data["sample_weight"].shape == (1,)
        assert data["sample_weight"][0] == pytest.approx(0.7 * 4)
        np.testing.assert_allclose(data["style_embedding"], mlops_env[3])
        assert data["face_features"].tolist() == pytest.approx(
            [1.4, 120.0, 130.0, 110.0, 0.92, 0.85]
        )
        assert data["skin_features"].tolist() == pytest.approx([45.0, 15.0])
        assert meta["source"] == "implicit"
        assert meta["signal_type"] == "synthesize"
        assert meta["rec_rank"] == 1
        assert meta["ml_rank"] == 2
        assert meta["propensity"] == pytest.approx(0.25)
        assert meta["hairstyle_id"] == 3
        assert meta["analysis_id"] == ANALYSIS_ID
        assert _head_signal(s3_store, 3) == "synthesize"

        data5, meta5 = _load(s3_store, 5)
        assert data5["ground_truth"].tolist() == [40.0]
        assert data5["sample_weight"][0] == pytest.approx(0.3 * 1.25)
        assert meta5["signal_type"] == "shown_not_chosen"
        assert meta5["rec_rank"] == 2  # 카드의 표시 순위
        assert _head_signal(s3_store, 5) == "shown_not_chosen"

        data7, meta7 = _load(s3_store, 7)
        assert data7["sample_weight"][0] == pytest.approx(0.3)  # propensity 없음 -> 1.0
        assert meta7["propensity"] == 1.0
        assert meta7["ml_rank"] is None

        assert _counts(s3_store) == (3, 3)

    def test_no_raw_identifiers_in_sample(self, s3_store):
        _record(s3_store, SIGNAL_SYNTHESIZE, hairstyle_id=3)
        _, meta = _load(s3_store, 3)
        assert "user_id" not in meta
        assert "device_id" not in meta
        assert "user-should-not-leak" not in json.dumps(meta)

    def test_layout_matches_save_feedback(self, s3_store):
        explicit = s3_store.save_feedback(
            analysis_id=ANALYSIS_ID,
            face_shape="계란형",
            skin_tone="봄웜",
            hairstyle_id=3,
            feedback="good",
            face_features=[1.4, 120.0, 130.0, 110.0, 0.92, 0.85],
            skin_features=[45.0, 15.0],
        )
        assert explicit["success"]
        explicit_key = [
            obj["Key"]
            for obj in s3_store.s3_client.list_objects_v2(
                Bucket=BUCKET, Prefix="feedback/pending/"
            )["Contents"]
            if "/implicit/" not in obj["Key"] and obj["Key"].endswith(".npz")
        ][0]
        ref = np.load(
            io.BytesIO(
                s3_store.s3_client.get_object(Bucket=BUCKET, Key=explicit_key)[
                    "Body"
                ].read()
            ),
            allow_pickle=False,
        )

        _record(s3_store, SIGNAL_SAVE, hairstyle_id=3)
        data, _ = _load(s3_store, 3)

        assert set(data.files) == set(ref.files) | {"sample_weight"}
        for name in ref.files:
            assert data[name].dtype.kind == ref[name].dtype.kind, name
            if name != "metadata":
                assert data[name].dtype == ref[name].dtype, name
            assert data[name].shape == ref[name].shape, name

    def test_stronger_signal_overwrites_without_counting(self, s3_store):
        _record(s3_store, SIGNAL_SYNTHESIZE, hairstyle_id=3)
        assert _counts(s3_store) == (3, 3)

        assert _record(s3_store, SIGNAL_SAVE, hairstyle_id=3)["reason"] == "overwritten"
        data, meta = _load(s3_store, 3)
        assert meta["signal_type"] == "save"
        assert data["ground_truth"].tolist() == [80.0]
        assert data["sample_weight"][0] == pytest.approx(1.0 * 4)
        assert _head_signal(s3_store, 3) == "save"

        assert _record(s3_store, SIGNAL_SHARE, hairstyle_id=3)["reason"] == "overwritten"
        assert _head_signal(s3_store, 3) == "share"
        assert _counts(s3_store) == (3, 3)

    def test_weaker_or_equal_signal_is_skipped(self, s3_store):
        _record(s3_store, SIGNAL_SHARE, hairstyle_id=3)
        assert _counts(s3_store) == (1, 1)

        for signal in (SIGNAL_SAVE, SIGNAL_SYNTHESIZE, SIGNAL_SHARE):
            result = _record(s3_store, signal, hairstyle_id=3)
            assert result["reason"] == "skipped"
            assert result["recorded"] is False
        _, meta = _load(s3_store, 3)
        assert meta["signal_type"] == "share"
        assert _head_signal(s3_store, 3) == "share"

    def test_synthesize_on_shown_not_chosen_card_upgrades_it(self, s3_store):
        _record(s3_store, SIGNAL_SYNTHESIZE, hairstyle_id=3)
        result = _record(s3_store, SIGNAL_SYNTHESIZE, hairstyle_id=5, rec_rank=2)

        assert result["reason"] == "overwritten"
        assert result["shown_not_chosen"] == 0
        _, meta5 = _load(s3_store, 5)
        assert meta5["signal_type"] == "synthesize"
        _, meta3 = _load(s3_store, 3)
        assert meta3["signal_type"] == "synthesize"  # 첫 선택은 강등되지 않는다
        _, meta7 = _load(s3_store, 7)
        assert meta7["signal_type"] == "shown_not_chosen"
        assert _counts(s3_store) == (3, 3)

    def test_shown_not_chosen_only_on_first_synthesize(self, s3_store):
        _record(s3_store, SIGNAL_SYNTHESIZE, hairstyle_id=3)
        # 학습 후 pending 이 processed 로 옮겨진 상황
        for key in _implicit_keys(s3_store):
            s3_store.s3_client.delete_object(Bucket=BUCKET, Key=key)

        result = _record(s3_store, SIGNAL_SYNTHESIZE, hairstyle_id=5)
        assert result["reason"] == "created"
        assert result["shown_not_chosen"] == 0
        assert _implicit_keys(s3_store) == [implicit_key(ANALYSIS_ID, 5)]

    def test_shown_not_chosen_does_not_touch_existing(self, s3_store):
        _record(s3_store, SIGNAL_SAVE, hairstyle_id=5)  # 저장이 먼저 기록된 카드
        result = _record(s3_store, SIGNAL_SYNTHESIZE, hairstyle_id=3)

        assert result["shown_not_chosen"] == 1  # 7 만
        assert _head_signal(s3_store, 5) == "save"
        assert _counts(s3_store) == (3, 3)

    def test_save_or_share_does_not_create_shown_not_chosen(self, s3_store):
        result = _record(s3_store, SIGNAL_SHARE, hairstyle_id=3)
        assert result["shown_not_chosen"] == 0
        assert _implicit_keys(s3_store) == [implicit_key(ANALYSIS_ID, 3)]

    def test_match_by_style_name_when_id_missing(self, s3_store):
        result = _record(s3_store, SIGNAL_SAVE, style_name="레이어드컷")
        assert result["recorded"] is True
        assert result["hairstyle_id"] == 3
        _, meta = _load(s3_store, 3)
        assert meta["rec_rank"] == 1  # 요청에 없으면 카드의 표시 순위

    def test_id_not_on_cards_falls_back_to_style_name(self, s3_store):
        result = _record(s3_store, SIGNAL_SAVE, hairstyle_id=11, style_name="웨이브 펌")
        assert result["hairstyle_id"] == 7

    def test_unknown_or_trending_card_is_skipped(self, s3_store):
        assert _record(s3_store, SIGNAL_SAVE, style_name="없는스타일")["reason"] == (
            "card_not_found"
        )
        assert _record(s3_store, SIGNAL_SYNTHESIZE, style_name="트렌드A")["reason"] == (
            "not_ml_card"
        )
        assert _implicit_keys(s3_store) == []
        assert _counts(s3_store) == (0, 0)

    def test_missing_analysis_or_features_is_skipped(self, s3_store):
        result = record_style_signal(
            ANALYSIS_ID,
            SIGNAL_SAVE,
            hairstyle_id=3,
            store=s3_store,
            analysis_getter=lambda _id: None,
        )
        assert result["reason"] == "analysis_not_found"

        no_features = _analysis()
        del no_features["mediapipe_ITA_value"]
        assert _record(s3_store, SIGNAL_SAVE, analysis=no_features, hairstyle_id=3)[
            "reason"
        ] == "no_features"
        assert _implicit_keys(s3_store) == []

    def test_falls_back_to_head_then_put_without_conditional_put(self, s3_store):
        from botocore.exceptions import ParamValidationError

        real_put = s3_store.s3_client.put_object
        calls = {"conditional": 0}

        def put_without_if_none_match(**kwargs):
            if "IfNoneMatch" in kwargs:
                calls["conditional"] += 1
                raise ParamValidationError(report="Unknown parameter IfNoneMatch")
            return real_put(**kwargs)

        with patch.object(s3_store.s3_client, "put_object", put_without_if_none_match):
            first = _record(s3_store, SIGNAL_SYNTHESIZE, hairstyle_id=3)
            second = _record(s3_store, SIGNAL_SYNTHESIZE, hairstyle_id=5)

        assert calls["conditional"] >= 1
        assert first["shown_not_chosen"] == 2
        assert second["shown_not_chosen"] == 0  # 표식이 head 경로로도 지켜진다
        assert _counts(s3_store) == (3, 3)


class TestSkips:
    def test_mlops_disabled_skips_without_touching_store(self, monkeypatch):
        monkeypatch.setenv("MLOPS_ENABLED", "false")
        store = MagicMock()
        getter = MagicMock()
        result = record_style_signal(
            ANALYSIS_ID, SIGNAL_SAVE, hairstyle_id=3, store=store, analysis_getter=getter
        )
        assert result["reason"] == "mlops_disabled"
        getter.assert_not_called()
        assert store.mock_calls == []

    def test_store_disabled_skips(self, monkeypatch):
        monkeypatch.setenv("MLOPS_ENABLED", "true")
        store = MagicMock(enabled=False)
        getter = MagicMock()
        result = record_style_signal(
            ANALYSIS_ID, SIGNAL_SAVE, store=store, analysis_getter=getter
        )
        assert result["reason"] == "store_disabled"
        getter.assert_not_called()

    def test_errors_never_propagate(self, monkeypatch):
        monkeypatch.setenv("MLOPS_ENABLED", "true")
        store = MagicMock(enabled=True)

        def boom(_id):
            raise RuntimeError("dynamodb down")

        result = record_style_signal(
            ANALYSIS_ID, SIGNAL_SAVE, store=store, analysis_getter=boom
        )
        assert result == {"recorded": False, "reason": "error"}

    def test_invalid_inputs(self, monkeypatch):
        monkeypatch.setenv("MLOPS_ENABLED", "true")
        getter = MagicMock()
        assert record_style_signal("../x", SIGNAL_SAVE, analysis_getter=getter)[
            "reason"
        ] == "invalid_analysis_id"
        # shown_not_chosen 은 외부에서 들어올 수 없다
        assert record_style_signal(
            ANALYSIS_ID, SIGNAL_SHOWN_NOT_CHOSEN, analysis_getter=getter
        )["reason"] == "invalid_signal"
        getter.assert_not_called()


# ========== 엔드포인트 ==========

STYLE_ACTION_URL = "/api/events/style-action"


@pytest.fixture
def client():
    from main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=False)
def no_rate_limits():
    from api.endpoints import events as events_module
    from api.endpoints import synthesis as synthesis_module

    previous = (events_module.limiter.enabled, synthesis_module.limiter.enabled)
    events_module.limiter.enabled = False
    synthesis_module.limiter.enabled = False
    yield
    events_module.limiter.enabled, synthesis_module.limiter.enabled = previous


def _style_action(**overrides):
    body = {
        "analysis_id": ANALYSIS_ID,
        "style_name": "레이어드 컷",
        "hairstyle_id": 3,
        "action": "save",
        "result_id": "abcdef0123456789",
        "device_id": "a1b2c3d4e5f60718",
    }
    body.update(overrides)
    return body


@pytest.mark.usefixtures("no_rate_limits")
class TestStyleActionEndpoint:
    def test_route_path_and_background_record(self, client):
        with patch("api.endpoints.events.record_style_signal") as record:
            response = client.post(STYLE_ACTION_URL, json=_style_action(action="share"))

        assert response.status_code == 200
        assert response.json() == {"success": True}
        record.assert_called_once_with(
            ANALYSIS_ID, "share", hairstyle_id=3, style_name="레이어드 컷"
        )

    def test_v2_prefix_is_not_mounted(self, client):
        response = client.post("/api/v2/events/style-action", json=_style_action())
        assert response.status_code in (404, 405)

    def test_nullable_fields(self, client):
        with patch("api.endpoints.events.record_style_signal") as record:
            response = client.post(
                STYLE_ACTION_URL,
                json=_style_action(hairstyle_id=None, result_id=None, device_id=None),
            )
        assert response.status_code == 200
        assert record.call_args.kwargs["hairstyle_id"] is None

    @pytest.mark.parametrize(
        "body",
        [
            _style_action(action="like"),
            _style_action(hairstyle_id="abc"),
            {k: v for k, v in _style_action().items() if k != "action"},
            {k: v for k, v in _style_action().items() if k != "analysis_id"},
            {k: v for k, v in _style_action().items() if k != "style_name"},
        ],
    )
    def test_schema_violation_is_422(self, client, body):
        with patch("api.endpoints.events.record_style_signal") as record:
            response = client.post(STYLE_ACTION_URL, json=body)
        assert response.status_code == 422
        record.assert_not_called()

    def test_200_even_when_recording_is_skipped(self, client, monkeypatch):
        # 실제 기록 함수 - MLOPS 꺼짐이라 AWS 를 건드리지 않고 생략된다
        monkeypatch.setenv("MLOPS_ENABLED", "false")
        response = client.post(STYLE_ACTION_URL, json=_style_action())
        assert response.status_code == 200
        assert response.json() == {"success": True}

    def test_unknown_analysis_id_format_still_200(self, client, monkeypatch):
        monkeypatch.setenv("MLOPS_ENABLED", "true")
        response = client.post(STYLE_ACTION_URL, json=_style_action(analysis_id="x/../y"))
        assert response.status_code == 200

    def test_broken_token_is_not_401(self, client):
        with patch("api.endpoints.events.record_style_signal"):
            response = client.post(
                STYLE_ACTION_URL,
                json=_style_action(),
                headers={"Authorization": "Bearer not-a-jwt"},
            )
        assert response.status_code == 200

    def test_log_has_no_raw_device_id(self, client):
        with patch("api.endpoints.events.record_style_signal"), patch(
            "api.endpoints.events.log_structured"
        ) as log:
            client.post(STYLE_ACTION_URL, json=_style_action())
        event, fields = log.call_args.args
        assert event == "style_action"
        assert "a1b2c3d4e5f60718" not in json.dumps(fields, ensure_ascii=False)
        assert fields["has_device_id"] is True


# ----- 합성 선택 필드 -----

CACHE_KEY = "e" * 64


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def synthesis_mocks():
    from core.jwt_auth import create_access_token

    storage = MagicMock()
    storage.enabled = True
    storage.build_cache_key.return_value = CACHE_KEY
    storage.get_cached_result.return_value = None
    storage.save_user_result.return_value = "https://s3/result.png"

    service = MagicMock()
    service.synthesize_hairstyle.return_value = {
        "success": True,
        "image_base64": "aW1n",
        "image_format": "png",
        "message": "적용되었습니다.",
        "api_calls": 1,
    }
    service.synthesize_with_reference.return_value = (
        service.synthesize_hairstyle.return_value
    )
    credit = MagicMock()
    credit.consume_for_synthesis.return_value = (4, "free")

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
        "api.endpoints.synthesis.get_user_repository"
    ), patch(
        "api.endpoints.synthesis.record_style_signal"
    ) as record, patch(
        "api.endpoints.synthesis.log_structured"
    ) as log:
        yield {
            "auth": {"Authorization": f"Bearer {create_access_token('implicit-user')}"},
            "storage": storage,
            "service": service,
            "record": record,
            "log": log,
        }


def _post_synthesize(client, auth, extra=None):
    data = {"hairstyle_name": "레이어드 컷", "gender": "male"}
    data.update(extra or {})
    return client.post(
        "/api/v2/synthesize",
        files={"file": ("face.png", _png_bytes(), "image/png")},
        data=data,
        headers=auth,
    )


def _success_logs(log):
    return [c.args[1] for c in log.call_args_list if c.args[0] == "synthesis_success"]


def _stable(body):
    return {k: v for k, v in body.items() if k != "processing_time"}


@pytest.mark.usefixtures("no_rate_limits")
class TestSynthesisSignalFields:
    def test_legacy_request_unchanged_and_not_recorded(self, client, synthesis_mocks):
        legacy = _post_synthesize(client, synthesis_mocks["auth"])
        assert legacy.status_code == 200
        synthesis_mocks["record"].assert_not_called()
        assert _success_logs(synthesis_mocks["log"])[-1]["analysis_id"] is None
        assert _success_logs(synthesis_mocks["log"])[-1]["rec_rank"] is None

        with_fields = _post_synthesize(
            client,
            synthesis_mocks["auth"],
            {"analysis_id": ANALYSIS_ID, "rec_rank": "2", "hairstyle_id": "5"},
        )
        assert with_fields.status_code == 200
        # 응답 본문은 필드 유무와 무관하게 같다
        assert _stable(with_fields.json()) == _stable(legacy.json())
        # 합성 서비스 호출 인자도 같다
        calls = synthesis_mocks["service"].synthesize_hairstyle.call_args_list
        assert calls[0] == calls[1]

    def test_fields_are_logged_and_recorded(self, client, synthesis_mocks):
        response = _post_synthesize(
            client,
            synthesis_mocks["auth"],
            {"analysis_id": ANALYSIS_ID, "rec_rank": "2", "hairstyle_id": "5"},
        )
        assert response.status_code == 200
        synthesis_mocks["record"].assert_called_once_with(
            ANALYSIS_ID,
            "synthesize",
            hairstyle_id=5,
            style_name="레이어드 컷",
            rec_rank=2,
        )
        log = _success_logs(synthesis_mocks["log"])[-1]
        assert log["analysis_id"] == ANALYSIS_ID
        assert log["rec_rank"] == 2

    def test_invalid_values_are_ignored_not_rejected(self, client, synthesis_mocks):
        response = _post_synthesize(
            client,
            synthesis_mocks["auth"],
            {"analysis_id": ANALYSIS_ID, "rec_rank": "nine", "hairstyle_id": "-3"},
        )
        assert response.status_code == 200
        kwargs = synthesis_mocks["record"].call_args.kwargs
        assert kwargs["rec_rank"] is None
        assert kwargs["hairstyle_id"] is None

        synthesis_mocks["record"].reset_mock()
        response = _post_synthesize(
            client,
            synthesis_mocks["auth"],
            {"analysis_id": "../../evil", "rec_rank": "1"},
        )
        assert response.status_code == 200
        synthesis_mocks["record"].assert_not_called()

    def test_cache_hit_is_recorded(self, client, synthesis_mocks):
        synthesis_mocks["storage"].get_cached_result.return_value = {
            "image_base64": "aW1n",
            "image_format": "png",
        }
        response = _post_synthesize(
            client, synthesis_mocks["auth"], {"analysis_id": ANALYSIS_ID}
        )
        assert response.status_code == 200
        assert response.json()["cached"] is True
        synthesis_mocks["record"].assert_called_once()
        assert _success_logs(synthesis_mocks["log"])[-1]["analysis_id"] == ANALYSIS_ID

    def test_failed_synthesis_is_not_recorded(self, client, synthesis_mocks):
        synthesis_mocks["service"].synthesize_hairstyle.return_value = {
            "success": False,
            "image_base64": None,
            "image_format": None,
            "message": "AI가 잠깐 헤맸어요.. 다시 시도해주세요!",
            "api_calls": 1,
        }
        response = _post_synthesize(
            client, synthesis_mocks["auth"], {"analysis_id": ANALYSIS_ID}
        )
        assert response.status_code == 422
        synthesis_mocks["record"].assert_not_called()

    def test_recording_failure_does_not_break_response(self, client, synthesis_mocks):
        # 실제 기록 함수를 쓰되 저장소가 터지는 상황 - 예외를 삼켜야 한다
        synthesis_mocks["record"].side_effect = lambda *a, **k: record_style_signal(
            *a, analysis_getter=_raise, store=MagicMock(enabled=True), **k
        )
        with patch.dict(os.environ, {"MLOPS_ENABLED": "true"}):
            response = _post_synthesize(
                client, synthesis_mocks["auth"], {"analysis_id": ANALYSIS_ID}
            )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_reference_endpoint_accepts_fields(self, client, synthesis_mocks):
        response = client.post(
            "/api/v2/synthesize-with-reference",
            files={
                "user_photo": ("face.png", _png_bytes(), "image/png"),
                "reference_photo": ("ref.png", _png_bytes(), "image/png"),
            },
            data={"gender": "male", "analysis_id": ANALYSIS_ID, "hairstyle_id": "3"},
            headers=synthesis_mocks["auth"],
        )
        assert response.status_code == 200
        synthesis_mocks["record"].assert_called_once_with(
            ANALYSIS_ID, "synthesize", hairstyle_id=3, style_name=None, rec_rank=None
        )


def _raise(_id):
    raise RuntimeError("boom")
