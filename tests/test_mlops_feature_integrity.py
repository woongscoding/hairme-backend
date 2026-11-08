"""
MLOps 학습 데이터 무결성 회귀 테스트 (2026-11-08 감사)

1. 측정값이 없는 피드백은 학습 샘플을 쓰지 않는다 (one-hot/기본값 폴백 제거)
2. feature_version 표식: 분석 행 + 모든 npz metadata, 트레이너 버전별 집계 + 필터
3. 임계값 트리거는 정기 실행과 같은 base + 전체 데이터 재학습 플래그를 보낸다
4. S3 모델 체크섬 검증은 fail closed (메타데이터/sha256 없으면 번들 모델)
5. 피부 영역 L*/b* 원값이 features 와 분석 행에 남는다

AWS 는 전부 목 처리한다.
"""

import hashlib
import io
import json
import logging
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import services.mlops.s3_feedback_store as sfs
from models.mediapipe_analyzer import (
    FEATURE_VERSION,
    FEATURE_VERSION_2D,
    FEATURE_VERSION_3D,
    MediaPipeFaceAnalyzer,
    MediaPipeFaceFeatures,
)
from services.mlops import feedback_features as ff

FACE = [1.22, 450.0, 560.0, 440.0, 0.80, 0.79]
SKIN = [78.0, 12.0]


def _import_lambda_trainer():
    sys.path.insert(0, "lambda_trainer")
    try:
        import lambda_function

        return lambda_function
    finally:
        sys.path.pop(0)


lf = _import_lambda_trainer()


# ========== 공용: 목 S3 store ==========


class FakeS3Client:
    METADATA_KEY = "feedback/metadata.json"

    def __init__(self):
        self.metadata = {"total_feedback_count": 0, "pending_count": 0}
        self.objects = {}

    def head_bucket(self, **kwargs):
        return {}

    def get_object(self, Bucket=None, Key=None, **kwargs):
        if Key == self.METADATA_KEY:
            return {"Body": io.BytesIO(json.dumps(self.metadata).encode("utf-8"))}
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket=None, Key=None, Body=None, **kwargs):
        if Key == self.METADATA_KEY:
            self.metadata = json.loads(Body)
        else:
            self.objects[Key] = Body
        return {}


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("MLOPS_ENABLED", "true")
    fake_s3 = FakeS3Client()
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = fake_s3
    monkeypatch.setattr(sfs, "boto3", mock_boto3)
    s = sfs.S3FeedbackStore(bucket_name="test-mlops-bucket")
    s.style_embeddings = np.ones((10, 384), dtype=np.float32)
    s.style_to_idx = {f"style_{i}": i for i in range(10)}
    assert s.enabled
    return s


def _npz_list(store):
    return [v for k, v in store.s3_client.objects.items() if k.endswith(".npz")]


def _meta(npz_bytes):
    data = np.load(io.BytesIO(npz_bytes), allow_pickle=False)
    return data, json.loads(str(data["metadata"][0]))


def _full_row(**overrides):
    row = {
        "analysis_id": "abcdef1234567890",
        "face_shape": "계란형",
        "personal_color": "봄웜",
        "mediapipe_face_ratio": 1.22,
        "mediapipe_forehead_width": 450.0,
        "mediapipe_cheekbone_width": 560.0,
        "mediapipe_jaw_width": 440.0,
        "mediapipe_forehead_ratio": 0.80,
        "mediapipe_jaw_ratio": 0.79,
        "mediapipe_ITA_value": 78.0,
        "mediapipe_hue_value": 12.0,
        "recommended_styles": [
            {"hairstyle_id": 3, "style_name": "style_3", "source": "ml"},
            {"hairstyle_id": None, "style_name": "허쉬컷", "source": "trending"},
        ],
    }
    row.update(overrides)
    return row


# ========== 1. 측정값 없으면 샘플을 쓰지 않는다 ==========


class TestNoGarbageSamples:
    @pytest.mark.parametrize(
        "face,skin",
        [
            (None, None),
            (FACE, None),
            (None, SKIN),
            (FACE[:5], SKIN),  # 차원 부족
            (FACE, [78.0, float("nan")]),  # 유한하지 않음
            ("123456", SKIN),
        ],
    )
    def test_store_skips_without_real_measurements(self, store, face, skin, caplog):
        with caplog.at_level(logging.WARNING, logger=sfs.logger.name):
            result = store.save_feedback(
                analysis_id="abcdef1234567890",
                face_shape="둥근형",
                skin_tone="봄웜",
                hairstyle_id=3,
                feedback="good",
                face_features=face,
                skin_features=skin,
            )
        assert result["success"] is False
        assert result["reason"] == "missing_features"
        assert _npz_list(store) == []
        assert store.s3_client.metadata["pending_count"] == 0
        logged = [json.loads(r.getMessage()) for r in caplog.records]
        assert any(
            e["event_type"] == "mlops_sample_skipped"
            and e["reason"] == "missing_features"
            for e in logged
        )

    def test_trending_store_skips_without_measurements(self, store):
        fake_recommender = MagicMock()
        fake_recommender.style_to_idx = {"허쉬컷": 0}
        fake_recommender.embeddings = np.zeros((1, 384), dtype=np.float32)
        with patch(
            "models.ml_recommender.get_ml_recommender", return_value=fake_recommender
        ):
            result = store.save_trending_feedback(
                analysis_id="abcdef1234567890",
                face_shape="둥근형",
                skin_tone="여름쿨",
                style_name="허쉬컷",
                feedback="good",
            )
        assert result["success"] is False
        assert result["reason"] == "missing_features"
        assert _npz_list(store) == []

    def test_label_encoders_removed(self):
        assert not hasattr(sfs.S3FeedbackStore, "_encode_face_shape")
        assert not hasattr(sfs.S3FeedbackStore, "_encode_skin_tone")

    def _run_export(self, monkeypatch, row, style_index=1):
        import database.dynamodb_connection as ddb

        monkeypatch.setenv("MLOPS_ENABLED", "true")
        fake_store = MagicMock()
        fake_store.save_feedback.return_value = {"success": True, "pending_count": 1}
        fake_store.save_trending_feedback.return_value = {"success": True}
        monkeypatch.setattr(ddb, "get_analysis", lambda _aid: row)
        with patch(
            "services.mlops.s3_feedback_store.get_s3_feedback_store",
            return_value=fake_store,
        ):
            ddb._save_feedback_to_s3(row["analysis_id"], style_index, "good")
        return fake_store

    @pytest.mark.parametrize(
        "missing", ["mediapipe_jaw_ratio", "mediapipe_hue_value", "mediapipe_ITA_value"]
    )
    def test_export_skips_row_missing_any_field(self, monkeypatch, caplog, missing):
        row = _full_row()
        row.pop(missing)
        with caplog.at_level(logging.WARNING):
            fake_store = self._run_export(monkeypatch, row)
        fake_store.save_feedback.assert_not_called()
        fake_store.save_trending_feedback.assert_not_called()
        events = []
        for record in caplog.records:
            try:
                events.append(json.loads(record.getMessage()))
            except ValueError:
                pass
        skipped = [e for e in events if e.get("event_type") == "mlops_sample_skipped"]
        assert skipped and skipped[0]["reason"] == "missing_mediapipe_features"
        assert skipped[0]["missing_fields"] == [missing]

    def test_export_skips_row_without_any_mediapipe(self, monkeypatch):
        row = {k: v for k, v in _full_row().items() if not k.startswith("mediapipe_")}
        fake_store = self._run_export(monkeypatch, row, style_index=2)
        fake_store.save_feedback.assert_not_called()
        fake_store.save_trending_feedback.assert_not_called()

    def test_export_passes_real_measurements_and_version(self, monkeypatch):
        row = _full_row(mediapipe_feature_version=FEATURE_VERSION_3D)
        fake_store = self._run_export(monkeypatch, row)
        kwargs = fake_store.save_feedback.call_args.kwargs
        assert kwargs["face_features"] == FACE
        assert kwargs["skin_features"] == SKIN
        assert kwargs["hairstyle_id"] == 3
        assert kwargs["feature_version"] == FEATURE_VERSION_3D

    def test_export_trending_uses_same_extraction(self, monkeypatch):
        fake_store = self._run_export(monkeypatch, _full_row(), style_index=2)
        kwargs = fake_store.save_trending_feedback.call_args.kwargs
        assert kwargs["face_features"] == FACE
        assert kwargs["skin_features"] == SKIN
        assert kwargs["feature_version"] == FEATURE_VERSION_2D

    def test_extract_features_rejects_non_finite(self):
        row = _full_row(mediapipe_jaw_width="nan")
        assert ff.extract_features(row) is None
        assert ff.missing_feature_fields(row) == ["mediapipe_jaw_width"]


# ========== 2. feature_version ==========


class TestFeatureVersionMarker:
    def test_constants(self):
        assert FEATURE_VERSION == FEATURE_VERSION_3D == "v2_3d"
        assert FEATURE_VERSION_2D == "v1_2d"
        # 트레이너는 서버 모듈을 import 하지 못하므로 문자열이 같아야 한다
        assert lf.FEATURE_VERSION_2D == FEATURE_VERSION_2D
        assert lf.FEATURE_VERSION_3D == FEATURE_VERSION_3D
        assert lf.FEATURE_VERSION_DEFAULT == FEATURE_VERSION_2D

    def test_feature_version_of_row(self):
        assert ff.feature_version_of({}) == "v1_2d"
        assert ff.feature_version_of({"mediapipe_head_yaw": 3.0}) == "v2_3d"
        assert ff.feature_version_of({"mediapipe_feature_version": "v2_3d"}) == "v2_3d"
        assert (
            ff.feature_version_of(
                {"mediapipe_feature_version": "v1_2d", "mediapipe_head_yaw": 1.0}
            )
            == "v1_2d"
        )
        assert ff.feature_version_of({"mediapipe_feature_version": " "}) == "v1_2d"

    def test_store_npz_metadata_has_feature_version(self, store):
        store.save_feedback(
            analysis_id="abcdef1234567890",
            face_shape="계란형",
            skin_tone="봄웜",
            hairstyle_id=3,
            feedback="good",
            face_features=FACE,
            skin_features=SKIN,
            feature_version="v2_3d",
        )
        store.save_feedback(
            analysis_id="abcdef1234567890",
            face_shape="계란형",
            skin_tone="봄웜",
            hairstyle_id=4,
            feedback="bad",
            face_features=FACE,
            skin_features=SKIN,
        )
        versions = sorted(_meta(b)[1]["feature_version"] for b in _npz_list(store))
        assert versions == ["v1_2d", "v2_3d"]

    def test_trending_npz_metadata_has_feature_version(self, store):
        fake_recommender = MagicMock()
        fake_recommender.style_to_idx = {"허쉬컷": 0}
        fake_recommender.embeddings = np.zeros((1, 384), dtype=np.float32)
        with patch(
            "models.ml_recommender.get_ml_recommender", return_value=fake_recommender
        ):
            result = store.save_trending_feedback(
                analysis_id="abcdef1234567890",
                face_shape="둥근형",
                skin_tone="여름쿨",
                style_name="허쉬컷",
                feedback="good",
                face_features=FACE,
                skin_features=SKIN,
                feature_version="v2_3d",
            )
        assert result["success"] is True
        data, meta = _meta(_npz_list(store)[0])
        assert meta["feature_version"] == "v2_3d"
        assert data["face_features"].tolist() == pytest.approx(FACE)

    def test_implicit_npz_metadata_has_feature_version(self):
        from services.mlops.implicit_signals import build_sample_npz

        for row, expected in (
            (_full_row(mediapipe_head_yaw=2.0), "v2_3d"),
            (_full_row(), "v1_2d"),
        ):
            body = build_sample_npz(
                analysis=row,
                analysis_id=row["analysis_id"],
                card=row["recommended_styles"][0],
                hairstyle_id=3,
                signal_type="synthesize",
                face_features=FACE,
                skin_features=SKIN,
                style_embedding=np.zeros(384, dtype=np.float32),
                rec_rank=1,
                timestamp=datetime(2026, 11, 8, tzinfo=timezone.utc),
            )
            assert _meta(body)[1]["feature_version"] == expected

    def _mp_features(self, **kw):
        base = dict(
            face_shape="계란형",
            skin_tone="봄웜",
            confidence=0.9,
            face_ratio=1.22,
            forehead_width=450.0,
            cheekbone_width=560.0,
            jaw_width=440.0,
            ITA_value=78.0,
            hue_value=12.0,
            face_features=FACE,
            skin_features=SKIN,
            head_yaw=3.0,
            head_pitch=-2.0,
            L_star=66.4,
            b_star=14.2,
        )
        base.update(kw)
        return MediaPipeFaceFeatures(**base)

    def _repo_save(self, mp_features):
        import database.dynamodb_repository as repo_mod

        captured = {}

        def fake_save(data):
            captured.update(data)
            return "id-1"

        with patch.object(repo_mod, "save_analysis", side_effect=fake_save):
            repo = repo_mod.DynamoDBAnalysisRepository()
            analysis_id = repo.save_analysis(
                image_hash="h",
                analysis_result={
                    "analysis": {"face_shape": "계란형", "personal_color": "봄웜"},
                    "recommendations": [],
                },
                processing_time=0.1,
                detection_method="mediapipe",
                mp_features=mp_features,
            )
        assert analysis_id == "id-1"
        return captured

    def test_repository_writes_version_and_lab(self):
        data = self._repo_save(self._mp_features())
        assert data["mediapipe_feature_version"] == "v2_3d"
        assert data["mediapipe_L_star"] == pytest.approx(66.4)
        assert data["mediapipe_b_star"] == pytest.approx(14.2)
        # 저장된 행은 feature_version_of 로 같은 값을 돌려준다
        assert ff.feature_version_of(data) == "v2_3d"

    def test_repository_skips_missing_lab(self):
        data = self._repo_save(self._mp_features(L_star=None, b_star=None))
        assert data["mediapipe_L_star"] is None
        assert data["mediapipe_b_star"] is None

    def test_save_analysis_persists_new_fields(self, monkeypatch):
        import database.dynamodb_connection as ddb

        table = MagicMock()
        monkeypatch.setattr(ddb, "_get_table", lambda: table)
        analysis_id = ddb.save_analysis(
            {
                "image_hash": "h",
                "face_shape": "계란형",
                "personal_color": "봄웜",
                "recommendations": [],
                "mediapipe_face_ratio": 1.22,
                "mediapipe_feature_version": "v2_3d",
                "mediapipe_L_star": 66.4,
                "mediapipe_b_star": 14.2,
            }
        )
        assert analysis_id
        item = table.put_item.call_args.kwargs["Item"]
        assert item["mediapipe_feature_version"] == "v2_3d"
        assert float(item["mediapipe_L_star"]) == pytest.approx(66.4)
        assert float(item["mediapipe_b_star"]) == pytest.approx(14.2)


# ========== 2b. 트레이너: 버전별 집계 + 필터 + 라벨 인코딩 제외 ==========


class _NoSuchKey(Exception):
    pass


class TrainerFakeS3:
    def __init__(self, objects):
        self.objects = dict(objects)
        self.exceptions = SimpleNamespace(NoSuchKey=_NoSuchKey)
        self.put_objects = {}

    def get_paginator(self, _name):
        outer = self

        class _P:
            def paginate(self, Bucket=None, Prefix=None, **kwargs):
                keys = sorted(k for k in outer.objects if k.startswith(Prefix))
                yield {"Contents": [{"Key": k} for k in keys]}

        return _P()

    def get_object(self, Bucket=None, Key=None, **kwargs):
        if Key in self.put_objects:
            body = self.put_objects[Key]
        elif Key in self.objects:
            body = self.objects[Key]
        else:
            raise _NoSuchKey(Key)
        if isinstance(body, str):
            body = body.encode("utf-8")
        return {"Body": io.BytesIO(body)}

    def put_object(self, Bucket=None, Key=None, Body=None, **kwargs):
        self.put_objects[Key] = Body
        return {}


def _sample(aid, hid, version=None, face=FACE, skin=SKIN):
    meta = {
        "analysis_id": aid,
        "hairstyle_id": hid,
        "feedback": "good",
        "timestamp": "2026-11-08T00:00:00+00:00",
    }
    if version is not None:
        meta["feature_version"] = version
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        face_features=np.array(face, dtype=np.float32),
        skin_features=np.array(skin, dtype=np.float32),
        style_embedding=np.zeros(384, dtype=np.float32),
        ground_truth=np.array([90.0], dtype=np.float32),
        metadata=np.array([json.dumps(meta)], dtype=str),
    )
    return buffer.getvalue()


ONEHOT_FACE = [0, 0, 0, 1, 0, 0]
ONEHOT_SKIN = [1, 0]


@pytest.fixture
def trainer_objects():
    return {
        "feedback/pending/a.npz": _sample("A", 1),  # 표식 없음 -> v1_2d
        "feedback/pending/b.npz": _sample("B", 1, "v1_2d"),
        "feedback/pending/c.npz": _sample("C", 1, "v2_3d"),
        "feedback/pending/d.npz": _sample("D", 1, "v2_3d"),
        "feedback/pending/e.npz": _sample(
            "E", 1, face=ONEHOT_FACE, skin=ONEHOT_SKIN
        ),  # 레거시 라벨 인코딩
    }


def _load(objects):
    with patch.object(lf, "get_s3_client", return_value=TrainerFakeS3(objects)):
        return lf.load_pending_feedbacks()


class TestTrainerFeatureVersions:
    def test_default_trains_all_versions_and_counts(self, monkeypatch, trainer_objects):
        monkeypatch.delenv("TRAIN_FEATURE_VERSIONS", raising=False)
        monkeypatch.delenv("TRAIN_DROP_LABEL_ENCODED", raising=False)
        face, _, _, _, count, file_keys, ids, info = _load(trainer_objects)

        assert count == 4
        assert sorted(ids) == ["A", "B", "C", "D"]
        assert len(file_keys) == 5  # 제외된 파일도 이동 대상
        assert lf.summarize_feature_versions(info["feature_versions"]) == {
            "v1_2d": 2,
            "v2_3d": 2,
        }
        assert info["filter"]["feature_versions_loaded"] == {"v1_2d": 3, "v2_3d": 2}
        assert info["filter"]["label_encoded_dropped"] == 1
        assert info["filter"]["excluded_by_feature_version"] == {}
        assert info["filter"]["allowed_feature_versions"] == ["v1_2d", "v2_3d"]
        assert not any(lf.is_label_encoded_sample(f, SKIN) for f in face)

    def test_env_filters_to_one_version(self, monkeypatch, trainer_objects):
        monkeypatch.setenv("TRAIN_FEATURE_VERSIONS", "v1_2d")
        _, _, _, _, count, file_keys, ids, info = _load(trainer_objects)
        assert count == 2
        assert sorted(ids) == ["A", "B"]
        assert len(file_keys) == 5
        assert info["filter"]["excluded_by_feature_version"] == {"v2_3d": 2}

        monkeypatch.setenv("TRAIN_FEATURE_VERSIONS", "v2_3d")
        _, _, _, _, count, _, ids, _ = _load(trainer_objects)
        assert sorted(ids) == ["C", "D"]

    def test_star_and_blank_mean_no_filter(self, monkeypatch):
        assert lf.allowed_feature_versions("*") is None
        assert lf.allowed_feature_versions("") == ["v1_2d", "v2_3d"]
        assert lf.allowed_feature_versions(" v2_3d , ") == ["v2_3d"]

    def test_label_encoded_drop_can_be_disabled(self, monkeypatch, trainer_objects):
        monkeypatch.setenv("TRAIN_DROP_LABEL_ENCODED", "false")
        _, _, _, _, count, _, _, info = _load(trainer_objects)
        assert count == 5
        assert info["filter"]["label_encoded_dropped"] == 0

    def test_is_label_encoded_sample(self):
        assert lf.is_label_encoded_sample(ONEHOT_FACE, ONEHOT_SKIN)
        assert lf.is_label_encoded_sample([1, 0, 0, 0, 0, 0], [0, 1])
        assert not lf.is_label_encoded_sample(FACE, SKIN)
        assert not lf.is_label_encoded_sample([0] * 6, [1, 0])  # 원핫 아님
        assert not lf.is_label_encoded_sample(ONEHOT_FACE, [78.0, 12.0])

    def test_everything_filtered_returns_empty(self, monkeypatch, trainer_objects):
        monkeypatch.setenv("TRAIN_FEATURE_VERSIONS", "v9_future")
        _, _, _, _, count, _, _, info = _load(trainer_objects)
        assert count == 0
        # 버전 필터가 먼저 적용된다 (라벨 인코딩 샘플도 표식 없음 = v1_2d)
        assert info["filter"]["excluded_by_feature_version"] == {
            "v1_2d": 3,
            "v2_3d": 2,
        }

    def test_complete_sample_info_defaults_version(self):
        info = lf.complete_sample_info(None, 2, ["k1", "k2"])
        assert info["feature_versions"] == ["v1_2d", "v1_2d"]

    def test_pipeline_report_has_version_counts(self, monkeypatch, trainer_objects):
        """학습 결과/리포트의 data_summary 에 전체·학습·홀드아웃 버전별 수"""
        monkeypatch.delenv("TRAIN_FEATURE_VERSIONS", raising=False)
        loaded = _load(trainer_objects)
        monkeypatch.setattr(lf, "load_pending_feedbacks", lambda **kw: loaded)
        monkeypatch.setattr(lf, "get_baseline_trained_at", lambda: (None, None))
        monkeypatch.setattr(lf, "MIN_SAMPLES", 1)
        # 시작점 모델이 없으면 학습 전에 멈춘다 - data_summary 는 그 전에 만들어진다
        monkeypatch.setattr(lf, "load_baseline_model", lambda fb: (None, "k"))
        monkeypatch.setattr(lf, "load_base_model", lambda **kw: (None, {}))
        result = lf.run_training_pipeline({"trigger_type": "manual", "force": True})

        summary = result["data_summary"]
        assert summary["feature_versions"] == {"v1_2d": 2, "v2_3d": 2}
        assert summary["base_model_feature_version"] == "v1_2d"
        assert summary["sample_filter"]["label_encoded_dropped"] == 1
        train = summary["train"]["feature_versions"]
        hold = summary["holdout"]["feature_versions"]
        for name in ("v1_2d", "v2_3d"):
            assert train.get(name, 0) + hold.get(name, 0) == 2


# ========== 3. 임계값 트리거 ==========


class TestThresholdTrigger:
    def _trigger(self, monkeypatch):
        import services.mlops.training_trigger as tt

        monkeypatch.setenv("MLOPS_ENABLED", "true")
        lambda_client = MagicMock()
        lambda_client.invoke.return_value = {"ResponseMetadata": {"RequestId": "r1"}}
        mock_boto3 = MagicMock()
        mock_boto3.client.side_effect = lambda name, **kw: (
            lambda_client if name == "lambda" else MagicMock()
        )
        monkeypatch.setattr(tt, "boto3", mock_boto3)
        trigger = tt.TrainingTrigger()
        assert trigger.enabled
        return tt, trigger, lambda_client

    def test_threshold_defers_to_weekly_schedule(self, monkeypatch):
        """임계치 도달은 기록만 - 게이트 거부 시 2시간마다 전체 재학습이 반복되지 않게"""
        tt, trigger, lambda_client = self._trigger(monkeypatch)
        result = trigger.check_and_trigger(tt.RETRAIN_THRESHOLD + 50)
        assert result["triggered"] is False
        assert result["reason"] == "threshold_reached_deferred_to_schedule"
        lambda_client.invoke.assert_not_called()

    def test_below_threshold_does_not_invoke(self, monkeypatch):
        tt, trigger, lambda_client = self._trigger(monkeypatch)
        result = trigger.check_and_trigger(tt.RETRAIN_THRESHOLD - 1)
        assert result["triggered"] is False
        lambda_client.invoke.assert_not_called()

    def test_manual_trigger_payload_unchanged(self, monkeypatch):
        _, trigger, lambda_client = self._trigger(monkeypatch)
        trigger.trigger_training("manual")
        payload = json.loads(lambda_client.invoke.call_args.kwargs["Payload"])
        assert payload["trigger_type"] == "manual"
        assert "from_base" not in payload and "include_processed" not in payload


# ========== 4. 체크섬 fail closed ==========


class TestChecksumFailClosed:
    def _s3_with_metadata(self, metadata):
        s3 = MagicMock()
        if metadata is None:
            s3.get_object.side_effect = Exception("NoSuchKey")
        else:
            s3.get_object.return_value = {
                "Body": io.BytesIO(json.dumps(metadata).encode("utf-8"))
            }
        return s3

    @pytest.mark.parametrize(
        "metadata",
        [None, {}, {"sha256": ""}, {"sha256": None}, {"version": "v6"}, ["x"]],
    )
    def test_missing_sha_fails(self, tmp_path, metadata):
        import models.ml_recommender as mlr

        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"model")
        s3 = self._s3_with_metadata(metadata)
        assert mlr._verify_model_checksum(s3, "b", str(model_file)) is False

    def test_trainer_written_metadata_passes_serving_check(self, tmp_path):
        """트레이너 writer 와 서빙 reader 의 키/필드 이름이 맞아야 한다"""
        import models.ml_recommender as mlr

        model_bytes = b"promoted-model-bytes"
        writes = {}
        trainer_s3 = MagicMock()
        trainer_s3.get_object.side_effect = Exception("NoSuchKey")
        trainer_s3.put_object.side_effect = lambda **kw: writes.update(
            {kw["Key"]: kw["Body"]}
        )
        with patch.object(lf, "get_s3_client", return_value=trainer_s3):
            assert lf.save_current_model_metadata(model_bytes, {}, "v6_x") is True

        metadata_key = "models/current/metadata.json"
        assert metadata_key in writes
        serving_s3 = MagicMock()
        serving_s3.get_object.return_value = {
            "Body": io.BytesIO(writes[metadata_key].encode("utf-8"))
        }
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(model_bytes)
        assert mlr._verify_model_checksum(serving_s3, "b", str(model_file)) is True
        assert serving_s3.get_object.call_args.kwargs["Key"] == metadata_key

    def test_download_without_sha_falls_back_and_removes_file(
        self, monkeypatch, tmp_path
    ):
        import models.ml_recommender as mlr

        monkeypatch.setenv("MLOPS_ENABLED", "true")
        monkeypatch.setattr(mlr.tempfile, "gettempdir", lambda: str(tmp_path))
        s3 = self._s3_with_metadata({"version": "v6_feedback_20260909"})
        s3.download_file.side_effect = lambda b, k, path: open(path, "wb").write(b"m")
        fake_boto3 = MagicMock()
        fake_boto3.client.return_value = s3
        monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

        assert mlr._download_current_model_from_s3() is None
        assert not (tmp_path / "hairme_model_current.pt").exists()

    def test_get_ml_recommender_uses_bundled_when_check_fails(self, monkeypatch):
        import models.ml_recommender as mlr

        monkeypatch.setattr(mlr, "_recommender_instance", None)
        monkeypatch.setattr(mlr, "_download_current_model_from_s3", lambda: None)
        created = []

        class FakeRecommender:
            def __init__(self, *args, **kwargs):
                created.append((args, kwargs))

        monkeypatch.setattr(mlr, "MLHairstyleRecommender", FakeRecommender)
        mlr.get_ml_recommender()
        # 인자 없이 = 기본 model_path (번들 v6)
        assert created == [((), {})]
        monkeypatch.setattr(mlr, "_recommender_instance", None)

    def test_bundled_default_path(self):
        import inspect

        import models.ml_recommender as mlr

        default = (
            inspect.signature(mlr.MLHairstyleRecommender.__init__)
            .parameters["model_path"]
            .default
        )
        assert default == "models/hairstyle_recommender_v6_multitoken.pt"

    def test_promotion_fails_when_metadata_write_fails(self):
        trainer_s3 = MagicMock()
        trainer_s3.get_object.side_effect = Exception("NoSuchKey")

        def put(**kw):
            if kw["Key"] == "models/current/metadata.json":
                raise RuntimeError("AccessDenied")
            return {}

        trainer_s3.put_object.side_effect = put
        with patch.object(lf, "get_s3_client", return_value=trainer_s3):
            assert lf.save_model_to_s3(lf.RecommendationModelV6(), {}, "v6_x") is False


# ========== 5. L*/b* ==========


class TestLabRawValues:
    def test_dataclass_defaults_and_to_dict(self):
        f = MediaPipeFaceFeatures(
            face_shape="계란형",
            skin_tone="봄웜",
            confidence=0.9,
            face_ratio=1.2,
            forehead_width=1.0,
            cheekbone_width=1.0,
            jaw_width=1.0,
            ITA_value=80.0,
            hue_value=12.0,
            face_features=FACE,
            skin_features=SKIN,
        )
        assert f.L_star is None and f.b_star is None
        assert f.feature_version == "v2_3d"
        d = f.to_dict()
        assert d["L_star"] is None and d["b_star"] is None
        assert d["feature_version"] == "v2_3d"

    def test_analyze_skin_tone_returns_lab_means(self):
        cv2 = pytest.importorskip("cv2")

        bgr = (150, 170, 210)  # 밝은 웜톤 피부색
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[:, :] = bgr

        n = 478
        rng = np.random.default_rng(0)
        landmark = [
            SimpleNamespace(x=float(x), y=float(y))
            for x, y in rng.uniform(0.3, 0.7, size=(n, 2))
        ]
        landmarks = SimpleNamespace(landmark=landmark)

        analyzer = object.__new__(MediaPipeFaceAnalyzer)  # mediapipe 초기화 생략
        result = analyzer._analyze_skin_tone(image, landmarks, (200, 200))
        assert len(result) == 6
        skin_tone, ita_legacy, hue, ita_corrected, L_star, b_star = result

        lab = cv2.cvtColor(image[:1, :1], cv2.COLOR_BGR2LAB)[0, 0]
        assert L_star == pytest.approx(float(lab[0]) * 100.0 / 255.0, abs=1e-6)
        assert b_star == pytest.approx(float(lab[2]) - 128.0, abs=1e-6)
        assert 0.0 <= L_star <= 100.0
        # ITA 보정값은 같은 L*/b* 로 계산된다
        import math

        assert ita_corrected == pytest.approx(
            math.atan((L_star - 50) / b_star) * 180 / math.pi, abs=1e-6
        )

    def test_empty_skin_region_has_no_lab(self):
        pytest.importorskip("cv2")
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        # 모든 랜드마크가 화면 밖 한 점 -> 마스크 비어 있음
        landmarks = SimpleNamespace(
            landmark=[SimpleNamespace(x=5.0, y=5.0) for _ in range(478)]
        )
        analyzer = object.__new__(MediaPipeFaceAnalyzer)
        result = analyzer._analyze_skin_tone(image, landmarks, (10, 10))
        assert result[-2:] == (None, None)
