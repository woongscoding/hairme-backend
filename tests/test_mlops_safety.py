"""
MLOps 보안/안정성 회귀 테스트

대상:
1. 안전한 역직렬화 (torch.load weights_only=True, np.load allow_pickle=False)
2. 피드백 NPZ 파일명 충돌 방지 (style index + uuid 접미사)
3. 재학습 트리거 스톰 방지 (임계값 교차 + 2시간 쿨다운)
4. S3 모델 SHA-256 무결성 검증
5. 랜덤 초기화 모델 배포 차단

AWS는 전부 목(mock) 처리하며 실제 네트워크 호출을 하지 않는다.
"""

import io
import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import services.mlops.s3_feedback_store as sfs

BUNDLED_MODEL_PATH = "models/hairstyle_recommender_v6_multitoken.pt"


# ========== Fake S3 ==========


class FakeS3Client:
    """save_feedback 경로에서 필요한 최소한의 S3 API만 구현한 목 클라이언트"""

    METADATA_KEY = "feedback/metadata.json"

    def __init__(self, metadata=None):
        self.metadata = metadata if metadata is not None else {}
        self.objects = {}
        self.put_calls = []

    def head_bucket(self, **kwargs):
        return {}

    def get_object(self, Bucket=None, Key=None, **kwargs):
        if Key == self.METADATA_KEY:
            body = json.dumps(self.metadata).encode("utf-8")
        elif Key in self.objects:
            body = self.objects[Key]
        else:
            raise KeyError(Key)
        return {"Body": io.BytesIO(body)}

    def put_object(self, Bucket=None, Key=None, Body=None, ContentType=None, **kwargs):
        self.put_calls.append(Key)
        if Key == self.METADATA_KEY:
            self.metadata = json.loads(Body)
        else:
            self.objects[Key] = Body
        return {}


def make_store(monkeypatch, metadata=None):
    """목 S3를 사용하는 S3FeedbackStore 생성"""
    monkeypatch.setenv("MLOPS_ENABLED", "true")

    default_metadata = {"total_feedback_count": 0, "pending_count": 0}
    fake_s3 = FakeS3Client(
        metadata=metadata if metadata is not None else default_metadata
    )

    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = fake_s3
    monkeypatch.setattr(sfs, "boto3", mock_boto3)

    store = sfs.S3FeedbackStore(bucket_name="test-mlops-bucket")

    # 임베딩 파일이 없는 테스트 환경이므로 직접 주입
    store.style_embeddings = np.arange(10 * 384, dtype=np.float32).reshape(10, 384)
    store.style_to_idx = {"style_%d" % i: i for i in range(10)}

    assert store.enabled is True
    return store, fake_s3


def npz_bodies(fake_s3):
    """put_object로 업로드된 NPZ (key, bytes) 목록"""
    return [(k, v) for k, v in fake_s3.objects.items() if k.endswith(".npz")]


# ========== (a) torch.load weights_only=True ==========


@pytest.mark.skipif(not os.path.exists(BUNDLED_MODEL_PATH), reason="번들 모델 없음")
def test_bundled_checkpoint_loads_with_weights_only_true():
    """번들 체크포인트가 weights_only=True로 로드되어야 한다"""
    checkpoint = torch.load(BUNDLED_MODEL_PATH, map_location="cpu", weights_only=True)

    assert isinstance(checkpoint, dict)
    assert "model_state_dict" in checkpoint
    assert isinstance(checkpoint.get("config"), dict)
    assert checkpoint["config"].get("version") == "v6"


@pytest.mark.skipif(not os.path.exists(BUNDLED_MODEL_PATH), reason="번들 모델 없음")
def test_real_loader_path_uses_weights_only_true(monkeypatch):
    """실제 로더(get_ml_recommender)가 번들 모델을 weights_only=True로 로드"""
    import models.ml_recommender as mlr

    # SentenceTransformer 다운로드 스킵 (Lambda 환경으로 인식시킴)
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "test-fn")
    monkeypatch.setenv("MLOPS_ENABLED", "false")

    # S3 다운로드 경로 차단 -> 번들 모델 사용
    monkeypatch.setattr(mlr, "_download_current_model_from_s3", lambda: None)
    monkeypatch.setattr(mlr, "_recommender_instance", None)

    original_load = torch.load
    seen = {}

    def spy_load(*args, **kwargs):
        seen["weights_only"] = kwargs.get("weights_only")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(mlr.torch, "load", spy_load)

    recommender = mlr.get_ml_recommender()

    assert seen["weights_only"] is True
    assert recommender.model_version == "v6"
    assert isinstance(recommender.model, mlr.RecommendationModelV6)


def test_mlops_modules_have_no_unsafe_deserialization():
    """수정 대상 파일에 weights_only=False / allow_pickle=True 가 남아있지 않아야 한다"""
    targets = [
        "models/ml_recommender.py",
        "lambda_trainer/lambda_function.py",
        "services/mlops/trainer_lambda.py",
        "services/mlops/s3_feedback_store.py",
        "services/mlops/metrics.py",
    ]
    for path in targets:
        with io.open(path, encoding="utf-8") as f:
            source = f.read()
        assert "weights_only=False" not in source, path
        assert "allow_pickle=True" not in source, path


# ========== (b) NPZ round-trip with allow_pickle=False ==========


def test_saved_npz_loads_without_pickle(monkeypatch):
    """store가 저장한 NPZ는 allow_pickle=False로 읽을 수 있어야 한다"""
    store, fake_s3 = make_store(monkeypatch)

    result = store.save_feedback(
        analysis_id="abcdef1234567890",
        face_shape="계란형",
        skin_tone="봄웜",
        hairstyle_id=3,
        feedback="good",
        face_features=[0.1] * 6,
        skin_features=[0.2] * 2,
    )

    assert result["success"] is True

    uploaded = npz_bodies(fake_s3)
    assert len(uploaded) == 1

    data = np.load(io.BytesIO(uploaded[0][1]), allow_pickle=False)

    assert set(data.files) == {
        "face_features",
        "skin_features",
        "style_embedding",
        "ground_truth",
        "metadata",
    }
    assert data["face_features"].shape == (6,)
    assert data["skin_features"].shape == (2,)
    assert data["style_embedding"].shape == (384,)
    assert float(data["ground_truth"][0]) == 90.0

    # object 배열이 하나라도 있으면 allow_pickle=False 로드가 불가능하다
    for name in data.files:
        assert data[name].dtype != np.dtype("O"), name

    meta = json.loads(str(data["metadata"][0]))
    assert meta["hairstyle_id"] == 3
    assert meta["feedback"] == "good"


def test_saved_trending_npz_loads_without_pickle(monkeypatch):
    """트렌드 피드백 NPZ도 allow_pickle=False로 읽을 수 있어야 한다"""
    store, fake_s3 = make_store(monkeypatch)

    fake_recommender = MagicMock()
    fake_recommender.sentence_model.encode.return_value = np.zeros(
        384, dtype=np.float32
    )

    with patch(
        "models.ml_recommender.get_ml_recommender", return_value=fake_recommender
    ):
        result = store.save_trending_feedback(
            analysis_id="abcdef1234567890",
            face_shape="둥근형",
            skin_tone="여름쿨",
            style_name="레이어드 컷",
            feedback="bad",
        )

    assert result["success"] is True

    uploaded = npz_bodies(fake_s3)
    assert len(uploaded) == 1

    data = np.load(io.BytesIO(uploaded[0][1]), allow_pickle=False)
    assert float(data["ground_truth"][0]) == 10.0
    for name in data.files:
        assert data[name].dtype != np.dtype("O"), name


# ========== (c) 파일명 충돌 방지 ==========


def test_feedback_filename_includes_style_and_is_unique(monkeypatch):
    """같은 analysis_id / 같은 날짜라도 파일명이 겹치지 않아야 한다"""
    store, fake_s3 = make_store(monkeypatch)

    analysis_id = "abcdef1234567890"

    for hairstyle_id in (3, 7):
        result = store.save_feedback(
            analysis_id=analysis_id,
            face_shape="계란형",
            skin_tone="봄웜",
            hairstyle_id=hairstyle_id,
            feedback="good",
        )
        assert result["success"] is True

    keys = [k for k, _ in npz_bodies(fake_s3)]
    assert len(keys) == 2, keys
    assert len(set(keys)) == 2

    assert any("_s3_" in k for k in keys)
    assert any("_s7_" in k for k in keys)
    assert all(k.startswith("feedback/pending/") for k in keys)
    assert all(analysis_id[:8] in k for k in keys)


def test_same_style_twice_still_unique(monkeypatch):
    """동일 스타일에 대한 중복 피드백도 uuid 접미사로 구분된다"""
    store, fake_s3 = make_store(monkeypatch)

    for _ in range(2):
        store.save_feedback(
            analysis_id="abcdef1234567890",
            face_shape="계란형",
            skin_tone="봄웜",
            hairstyle_id=3,
            feedback="good",
        )

    keys = [k for k, _ in npz_bodies(fake_s3)]
    assert len(keys) == 2
    assert len(set(keys)) == 2


# ========== (d) 재학습 트리거 스톰 방지 ==========


def test_trigger_fires_on_threshold_crossing(monkeypatch):
    """99 -> 100 으로 임계값을 넘는 순간 트리거된다"""
    monkeypatch.setattr(sfs, "RETRAIN_THRESHOLD", 100)

    store, fake_s3 = make_store(
        monkeypatch, metadata={"total_feedback_count": 99, "pending_count": 99}
    )

    result = store.save_feedback(
        analysis_id="abcdef1234567890",
        face_shape="계란형",
        skin_tone="봄웜",
        hairstyle_id=1,
        feedback="good",
    )

    assert result["pending_count"] == 100
    assert result["should_trigger_training"] is True
    assert fake_s3.metadata.get("training_triggered_at")


def test_trigger_does_not_fire_again_within_cooldown(monkeypatch):
    """100 -> 101 은 2시간 내에 재트리거되지 않는다"""
    monkeypatch.setattr(sfs, "RETRAIN_THRESHOLD", 100)

    now = datetime.now(timezone.utc)
    store, _fake_s3 = make_store(
        monkeypatch,
        metadata={
            "total_feedback_count": 100,
            "pending_count": 100,
            "training_triggered_at": (now - timedelta(minutes=30)).isoformat(),
        },
    )

    result = store.save_feedback(
        analysis_id="abcdef1234567890",
        face_shape="계란형",
        skin_tone="봄웜",
        hairstyle_id=1,
        feedback="good",
    )

    assert result["pending_count"] == 101
    assert result["should_trigger_training"] is False


def test_trigger_fires_again_after_cooldown(monkeypatch):
    """마지막 트리거로부터 2시간이 지나면 재트리거된다"""
    monkeypatch.setattr(sfs, "RETRAIN_THRESHOLD", 100)

    now = datetime.now(timezone.utc)
    store, _fake_s3 = make_store(
        monkeypatch,
        metadata={
            "total_feedback_count": 100,
            "pending_count": 100,
            "training_triggered_at": (now - timedelta(hours=3)).isoformat(),
        },
    )

    result = store.save_feedback(
        analysis_id="abcdef1234567890",
        face_shape="계란형",
        skin_tone="봄웜",
        hairstyle_id=1,
        feedback="good",
    )

    assert result["pending_count"] == 101
    assert result["should_trigger_training"] is True


def test_should_trigger_training_helper_matrix(monkeypatch):
    """_should_trigger_training 단위 검증"""
    monkeypatch.setattr(sfs, "RETRAIN_THRESHOLD", 100)
    store, _fake_s3 = make_store(monkeypatch)

    now = datetime.now(timezone.utc)

    # 임계값 미만
    meta = {}
    assert store._should_trigger_training(meta, 98, 99, now) is False
    assert "training_triggered_at" not in meta

    # 교차 순간
    meta = {}
    assert store._should_trigger_training(meta, 99, 100, now) is True
    assert meta["training_triggered_at"] == now.isoformat()

    # 쿨다운 내 재발동 없음
    assert store._should_trigger_training(meta, 100, 101, now) is False

    # 쿨다운 경과 후 재발동
    later = now + timedelta(hours=2, minutes=1)
    assert store._should_trigger_training(meta, 101, 102, later) is True
    assert meta["training_triggered_at"] == later.isoformat()

    # training_triggered_at 이 없으면 재발동
    meta2 = {}
    assert store._should_trigger_training(meta2, 150, 151, now) is True


# ========== 모델 무결성 검증 ==========


def test_verify_model_checksum_detects_mismatch(tmp_path):
    """metadata.json 의 sha256과 불일치하면 검증 실패"""
    import models.ml_recommender as mlr

    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"hello-model")

    s3 = MagicMock()
    s3.get_object.return_value = {
        "Body": io.BytesIO(json.dumps({"sha256": "deadbeef"}).encode("utf-8"))
    }

    assert mlr._verify_model_checksum(s3, "bucket", str(model_file)) is False


def test_verify_model_checksum_accepts_match(tmp_path):
    """sha256이 일치하면 검증 통과"""
    import hashlib

    import models.ml_recommender as mlr

    payload = b"hello-model"
    model_file = tmp_path / "model.pt"
    model_file.write_bytes(payload)

    s3 = MagicMock()
    s3.get_object.return_value = {
        "Body": io.BytesIO(
            json.dumps({"sha256": hashlib.sha256(payload).hexdigest()}).encode("utf-8")
        )
    }

    assert mlr._verify_model_checksum(s3, "bucket", str(model_file)) is True


def test_verify_model_checksum_passes_when_metadata_missing(tmp_path):
    """metadata.json 이 없으면 경고 후 통과"""
    import models.ml_recommender as mlr

    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"x")

    s3 = MagicMock()
    s3.get_object.side_effect = Exception("NoSuchKey")

    assert mlr._verify_model_checksum(s3, "bucket", str(model_file)) is True


# ========== Trainer: 랜덤 초기화 모델 배포 차단 ==========


def _import_lambda_trainer():
    import sys

    sys.path.insert(0, "lambda_trainer")
    try:
        import lambda_function

        return lambda_function
    finally:
        sys.path.pop(0)


def test_load_base_model_aborts_without_allow_random_init():
    """models/current/model.pt 가 없으면 랜덤 초기화 대신 중단해야 한다"""
    lf = _import_lambda_trainer()

    class NoSuchKey(Exception):
        pass

    fake_s3 = MagicMock()
    fake_s3.exceptions.NoSuchKey = NoSuchKey
    fake_s3.get_object.side_effect = NoSuchKey()

    with patch.object(lf, "get_s3_client", return_value=fake_s3):
        model, config = lf.load_base_model()
        assert model is None
        assert config == {}

        model, config = lf.load_base_model(allow_random_init=True)
        assert model is not None
        assert config["version"] == "v6"


def test_save_model_to_s3_writes_sha256_metadata():
    """save_model_to_s3 가 models/current/metadata.json 에 sha256을 기록한다"""
    import hashlib

    lf = _import_lambda_trainer()

    puts = {}

    fake_s3 = MagicMock()
    fake_s3.get_object.side_effect = Exception("NoSuchKey")

    def fake_put(Bucket=None, Key=None, Body=None, ContentType=None, **kwargs):
        puts[Key] = Body
        return {}

    fake_s3.put_object.side_effect = fake_put

    model = lf.RecommendationModelV6()

    with patch.object(lf, "get_s3_client", return_value=fake_s3):
        assert lf.save_model_to_s3(model, {"version": "v6"}, "v6_test") is True

    assert "models/current/model.pt" in puts
    assert "models/current/metadata.json" in puts

    metadata = json.loads(puts["models/current/metadata.json"])
    expected = hashlib.sha256(puts["models/current/model.pt"]).hexdigest()
    assert metadata["sha256"] == expected
    assert metadata["version"] == "v6_test"
