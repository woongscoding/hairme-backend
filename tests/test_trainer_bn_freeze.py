"""
Trainer Lambda BatchNorm 고정 / base 재학습 모드 회귀 테스트

대상: lambda_trainer/lambda_function.py

검증 항목:
1. fine-tuning 중 BatchNorm running_mean/running_var/num_batches_tracked 가 갱신되지 않음
   (Linear 및 BN affine 파라미터는 계속 학습됨)
2. from_base=True → models/base/model.pt 에서 시작, 없으면 명확한 메시지로 중단
3. include_processed=True → feedback/processed/ 데이터도 학습에 포함,
   단 processed/ 파일은 이동하지 않음
4. S3 나열은 반드시 paginator 사용 (list_objects_v2 직접 호출 금지)
5. from_base / include_processed / source key 가 config·metadata·결과 본문에 기록됨

AWS 호출은 전부 목(mock) 처리한다.
"""

import io
import json
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


def _import_lambda_trainer():
    """lambda_trainer/lambda_function.py 를 저장소 import 없이 로드"""
    sys.path.insert(0, "lambda_trainer")
    try:
        import lambda_function

        return lambda_function
    finally:
        sys.path.pop(0)


lf = _import_lambda_trainer()


# ========== 공용 목 ==========


class _NoSuchKey(Exception):
    pass


class _S3Exceptions:
    NoSuchKey = _NoSuchKey


class FakeS3:
    """paginator / get_object / copy_object 만 구현한 최소 목 S3"""

    def __init__(self, keys_by_prefix=None, objects=None):
        self.keys_by_prefix = keys_by_prefix or {}
        self.objects = objects or {}
        self.exceptions = _S3Exceptions
        self.copied = []
        self.deleted = []
        self.put_objects = {}
        self.list_objects_v2_calls = 0

    # 페이지네이터를 쓰지 않으면 테스트가 실패하도록 강제
    def list_objects_v2(self, **kwargs):  # pragma: no cover - 호출되면 실패
        self.list_objects_v2_calls += 1
        raise AssertionError(
            "list_objects_v2 를 직접 호출하면 1000개 초과 키가 누락된다 - "
            "get_paginator 를 사용해야 한다"
        )

    def get_paginator(self, operation_name):
        assert operation_name == "list_objects_v2"
        outer = self

        class _Paginator:
            def paginate(self, Bucket=None, Prefix=None, **kwargs):
                keys = outer.keys_by_prefix.get(Prefix, [])
                # 페이지네이션 동작 확인용으로 2개씩 쪼개서 반환
                if not keys:
                    yield {}
                    return
                for start in range(0, len(keys), 2):
                    yield {"Contents": [{"Key": k} for k in keys[start : start + 2]]}

        return _Paginator()

    def get_object(self, Bucket=None, Key=None, **kwargs):
        if Key not in self.objects:
            raise _NoSuchKey(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket=None, Key=None, Body=None, **kwargs):
        self.put_objects[Key] = Body
        return {}

    def copy_object(self, Bucket=None, CopySource=None, Key=None, **kwargs):
        self.copied.append((CopySource["Key"], Key))
        return {}

    def delete_object(self, Bucket=None, Key=None, **kwargs):
        self.deleted.append(Key)
        return {}


def _npz_bytes(gt=80.0):
    """피드백 NPZ 1건 직렬화"""
    buffer = io.BytesIO()
    np.savez(
        buffer,
        face_features=np.array(
            [1.2, 100.0, 130.0, 105.0, 0.78, 0.80], dtype=np.float32
        ),
        skin_features=np.array([70.0, 15.0], dtype=np.float32),
        style_embedding=np.zeros(384, dtype=np.float32),
        ground_truth=np.array([gt], dtype=np.float32),
    )
    return buffer.getvalue()


def _checkpoint_bytes(model=None):
    """번들 v6 체크포인트와 동일한 레이아웃의 체크포인트 직렬화"""
    model = model or lf.RecommendationModelV6()
    buffer = io.BytesIO()
    torch.save(
        {
            "epoch": 42,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "best_val_loss": 0.0123,
            "history": {"train_loss": [0.5, 0.4]},
            "config": {
                "version": "v6",
                "face_feat_dim": 6,
                "skin_feat_dim": 2,
                "style_embed_dim": 384,
                "token_dim": 128,
                "num_heads": 4,
                "normalized": True,
                "label_min": 10.0,
                "label_max": 95.0,
                "label_range": 85.0,
                "attention_type": "multi_token",
            },
        },
        buffer,
    )
    return buffer.getvalue()


# ========== (1) BatchNorm 고정 ==========


def test_set_batchnorm_eval_freezes_only_batchnorm():
    """set_batchnorm_eval 은 BatchNorm만 eval 로 두고 나머지는 train 유지"""
    model = lf.RecommendationModelV6()
    model.train()

    frozen = lf.set_batchnorm_eval(model)

    bn_modules = [
        m
        for m in model.modules()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
    ]
    assert frozen == len(bn_modules)
    assert frozen > 0

    for m in bn_modules:
        assert m.training is False

    # Dropout 등 나머지 모듈은 여전히 train 모드
    dropouts = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    assert dropouts and all(m.training for m in dropouts)


def test_fine_tune_does_not_update_batchnorm_running_stats():
    """2 스텝 학습 후 running stats/num_batches_tracked 는 그대로, 가중치는 변경"""
    torch.manual_seed(20260909)
    model = lf.RecommendationModelV6()

    # BN running stats 를 학습 시점의 값으로 흉내내기 (기본 0/1 이 아닌 값)
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.running_mean.add_(0.37)
            m.running_var.mul_(2.5)
            m.num_batches_tracked.fill_(1234)

    before_stats = {
        name: buf.detach().clone()
        for name, buf in model.named_buffers()
        if any(
            name.endswith(suffix)
            for suffix in ("running_mean", "running_var", "num_batches_tracked")
        )
    }
    assert before_stats

    before_linear = model.fc1.weight.detach().clone()
    before_bn_affine = model.bn1.weight.detach().clone()

    # drop_last=True 로 32개 배치 2스텝이 되도록 80 샘플
    n = 80
    rng = np.random.default_rng(7)
    face = np.stack(
        [
            np.array([1.2, 100.0, 130.0, 105.0, 0.78, 0.80], dtype=np.float32)
            + rng.normal(0, 0.01, 6).astype(np.float32)
            for _ in range(n)
        ]
    )
    skin = np.tile(np.array([70.0, 15.0], dtype=np.float32), (n, 1))
    style = rng.normal(0, 1, (n, 384)).astype(np.float32)
    gt = rng.uniform(10.0, 95.0, n).astype(np.float32)

    trained, stats = lf.fine_tune_model(model, face, skin, style, gt, epochs=1, lr=0.01)

    after_stats = dict(trained.named_buffers())
    for name, before in before_stats.items():
        torch.testing.assert_close(
            after_stats[name], before, msg=f"BatchNorm buffer 변경됨: {name}"
        )

    # 실제로 학습이 수행되었는지 (가중치 변화)
    assert not torch.equal(trained.fc1.weight.detach(), before_linear)
    # BN affine 파라미터는 계속 학습되어야 한다
    assert not torch.equal(trained.bn1.weight.detach(), before_bn_affine)

    assert stats["batchnorm_frozen"] > 0
    assert stats["samples"] == n


# ========== (2) from_base 모드 ==========


def test_load_base_model_uses_base_key_when_from_base():
    """from_base=True 이면 models/base/model.pt 를 읽는다"""
    fake_s3 = FakeS3(objects={lf.BASE_MODEL_KEY: _checkpoint_bytes()})

    with patch.object(lf, "get_s3_client", return_value=fake_s3):
        model, config = lf.load_base_model(from_base=True)

    assert model is not None
    assert isinstance(model, lf.RecommendationModelV6)
    # 번들 체크포인트 config 키가 그대로 살아있어야 한다
    for key in ("token_dim", "num_heads", "label_min", "label_max", "attention_type"):
        assert key in config
    assert config["token_dim"] == 128


def test_load_base_model_defaults_fill_missing_config_keys():
    """config 에 일부 키가 없어도 기본값으로 채워 로드된다"""
    model = lf.RecommendationModelV6()
    buffer = io.BytesIO()
    torch.save({"model_state_dict": model.state_dict(), "config": {}}, buffer)
    fake_s3 = FakeS3(objects={lf.BASE_MODEL_KEY: buffer.getvalue()})

    with patch.object(lf, "get_s3_client", return_value=fake_s3):
        loaded, config = lf.load_base_model(from_base=True)

    assert loaded is not None
    assert config["token_dim"] == 128
    assert config["num_heads"] == 4
    assert config["label_range"] == lf.LABEL_RANGE


def test_load_base_model_missing_base_aborts_even_with_allow_random_init():
    """base 키가 없으면 allow_random_init 이어도 랜덤 모델을 만들지 않는다"""
    fake_s3 = FakeS3(objects={})

    with patch.object(lf, "get_s3_client", return_value=fake_s3):
        model, config = lf.load_base_model(allow_random_init=True, from_base=True)

    assert model is None
    assert config == {}


def test_get_source_model_key():
    assert lf.get_source_model_key(True) == "models/base/model.pt"
    assert lf.get_source_model_key(False) == "models/current/model.pt"


# ========== (3)/(4) include_processed + paginator ==========


def test_load_feedbacks_uses_paginator_and_pending_only_by_default():
    """기본값은 pending/ 만, 그리고 반드시 paginator 로 나열"""
    pending = [f"feedback/pending/p{i}.npz" for i in range(5)]
    processed = [f"feedback/processed/batch_a/q{i}.npz" for i in range(3)]
    objects = {k: _npz_bytes() for k in pending + processed}
    fake_s3 = FakeS3(
        keys_by_prefix={
            lf.PENDING_PREFIX: pending,
            lf.PROCESSED_PREFIX: processed,
        },
        objects=objects,
    )

    with patch.object(lf, "get_s3_client", return_value=fake_s3):
        face, skin, style, gt, count, keys, analysis_ids = lf.load_pending_feedbacks()

    assert fake_s3.list_objects_v2_calls == 0
    assert count == 5
    assert set(keys) == set(pending)
    assert face.shape == (5, 6)
    assert gt.shape == (5,)
    # 홀드아웃 분할용 analysis_id 가 샘플 수만큼 함께 나와야 한다
    assert len(analysis_ids) == 5


def test_load_feedbacks_include_processed():
    """include_processed=True 이면 processed/ 도 함께 로드"""
    pending = [f"feedback/pending/p{i}.npz" for i in range(5)]
    processed = [f"feedback/processed/batch_a/q{i}.npz" for i in range(3)]
    objects = {k: _npz_bytes() for k in pending + processed}
    fake_s3 = FakeS3(
        keys_by_prefix={
            lf.PENDING_PREFIX: pending,
            lf.PROCESSED_PREFIX: processed,
        },
        objects=objects,
    )

    with patch.object(lf, "get_s3_client", return_value=fake_s3):
        face, skin, style, gt, count, keys, analysis_ids = lf.load_pending_feedbacks(
            include_processed=True
        )

    assert count == 8
    assert set(keys) == set(pending) | set(processed)
    assert face.shape == (8, 6)
    assert len(analysis_ids) == 8


def test_move_pending_to_processed_skips_processed_keys():
    """processed/ 키가 섞여 들어와도 이동 대상은 pending/ 뿐"""
    fake_s3 = FakeS3()
    keys = [
        "feedback/pending/a.npz",
        "feedback/processed/batch_a/b.npz",
        "feedback/pending/c.npz",
    ]

    with patch.object(lf, "get_s3_client", return_value=fake_s3):
        assert lf.move_pending_to_processed(keys, "batch_test") is True

    assert fake_s3.deleted == ["feedback/pending/a.npz", "feedback/pending/c.npz"]
    assert [dst for _, dst in fake_s3.copied] == [
        "feedback/processed/batch_test/a.npz",
        "feedback/processed/batch_test/c.npz",
    ]


# ========== (5) 파이프라인 기록 ==========


def _fake_feedbacks(pending_count: int, processed_count: int = 0):
    total = pending_count + processed_count
    face = np.tile(
        np.array([1.2, 100.0, 130.0, 105.0, 0.78, 0.80], dtype=np.float32), (total, 1)
    )
    skin = np.tile(np.array([70.0, 15.0], dtype=np.float32), (total, 1))
    style = np.zeros((total, 384), dtype=np.float32)
    gt = np.array([90.0 if i % 2 == 0 else 10.0 for i in range(total)], np.float32)
    keys = [f"feedback/pending/f{i}.npz" for i in range(pending_count)] + [
        f"feedback/processed/batch_a/g{i}.npz" for i in range(processed_count)
    ]
    analysis_ids = [f"analysis-{i}" for i in range(total)]
    return face, skin, style, gt, total, keys, analysis_ids


def _patched_pipeline(pending_count, processed_count=0, saved=None):
    data = _fake_feedbacks(pending_count, processed_count)
    model = MagicMock()

    def _save(m, config, version):
        if saved is not None:
            saved["config"] = dict(config)
            saved["version"] = version
        return True

    return [
        patch.object(lf, "load_pending_feedbacks", return_value=data),
        patch.object(lf, "load_base_model", return_value=(model, {"version": "v6"})),
        # 학습 전/후 동일 지표 -> 품질 게이트 통과 (게이트 자체는 별도 테스트에서 검증)
        patch.object(
            lf,
            "evaluate_holdout",
            return_value={"mse": 0.1, "ranking_accuracy": 0.8, "num_pairs": 10},
        ),
        patch.object(lf, "fine_tune_model", return_value=(model, {"final_loss": 0.1})),
        patch.object(lf, "backup_lambda_config", return_value={}),
        patch.object(lf, "save_model_to_s3", side_effect=_save),
        patch.object(lf, "save_rejected_model", return_value=True),
        patch.object(lf, "move_pending_to_processed", return_value=True),
        patch.object(lf, "update_analyze_lambda_envvars", return_value=True),
        patch.object(lf, "update_metadata", return_value=None),
        patch.object(lf, "save_evaluation_report", return_value=True),
    ]


def test_pipeline_from_base_records_flags_and_moves_pending_only():
    """from_base + include_processed 기록, processed 파일은 이동 대상에서 제외"""
    saved = {}
    # 홀드아웃(15%)이 MIN_HOLDOUT_SAMPLES 를 넘도록 충분한 표본을 쓴다
    patches = _patched_pipeline(80, processed_count=160, saved=saved)
    for p in patches:
        p.start()
    try:
        result = lf.run_training_pipeline(
            {
                "trigger_type": "manual",
                "force": True,
                "from_base": True,
                "include_processed": True,
            }
        )
        # 시작점 모델을 base 로 요청했는지
        _, kwargs = lf.load_base_model.call_args
        assert kwargs["from_base"] is True
        # 로더에 include_processed 전달
        _, load_kwargs = lf.load_pending_feedbacks.call_args
        assert load_kwargs["include_processed"] is True
        # pending 80건만 이동
        moved_keys = lf.move_pending_to_processed.call_args[0][0]
        assert len(moved_keys) == 80
        assert all(k.startswith(lf.PENDING_PREFIX) for k in moved_keys)
    finally:
        for p in patches:
            p.stop()

    assert result["success"] is True
    assert result["samples_trained"] == 240
    assert result["from_base"] is True
    assert result["include_processed"] is True
    assert result["source_model_key"] == lf.BASE_MODEL_KEY
    assert result["pending_files_moved"] == 80
    # 홀드아웃은 학습에 쓰이지 않는다
    assert result["train_size"] + result["holdout_size"] == 240
    assert saved["config"]["train_samples"] == result["train_size"]

    # 저장되는 config 에도 기록
    assert saved["config"]["from_base"] is True
    assert saved["config"]["include_processed"] is True
    assert saved["config"]["source_model_key"] == lf.BASE_MODEL_KEY
    assert saved["config"]["batchnorm_frozen"] is True


def test_pipeline_defaults_are_current_model():
    """플래그가 없으면 기존과 동일하게 models/current/model.pt 사용"""
    patches = _patched_pipeline(200)
    for p in patches:
        p.start()
    try:
        result = lf.run_training_pipeline({"trigger_type": "scheduled"})
        _, kwargs = lf.load_base_model.call_args
        assert kwargs["from_base"] is False
        _, load_kwargs = lf.load_pending_feedbacks.call_args
        assert load_kwargs["include_processed"] is False
    finally:
        for p in patches:
            p.stop()

    assert result["success"] is True
    assert result["from_base"] is False
    assert result["source_model_key"] == lf.CURRENT_MODEL_KEY


def test_pipeline_from_base_missing_base_aborts_with_message():
    """base 모델이 없으면 명확한 메시지로 중단"""
    patches = _patched_pipeline(60)
    patches[1] = patch.object(lf, "load_base_model", return_value=(None, {}))
    for p in patches:
        p.start()
    try:
        result = lf.run_training_pipeline(
            {"trigger_type": "manual", "force": True, "from_base": True}
        )
    finally:
        for p in patches:
            p.stop()

    assert result["success"] is False
    assert result["message"] == "base model missing: upload models/base/model.pt"


def test_metadata_records_retrain_source():
    """save_current_model_metadata 가 from_base/source key 를 기록"""
    fake_s3 = FakeS3()
    config = {
        "version": "v6_feedback_20260909",
        "from_base": True,
        "include_processed": True,
        "source_model_key": lf.BASE_MODEL_KEY,
        "batchnorm_frozen": True,
    }

    with patch.object(lf, "get_s3_client", return_value=fake_s3):
        assert (
            lf.save_current_model_metadata(b"model-bytes", config, "v6_feedback_x")
            is True
        )

    metadata = json.loads(fake_s3.put_objects["models/current/metadata.json"])
    assert metadata["from_base"] is True
    assert metadata["include_processed"] is True
    assert metadata["source_model_key"] == lf.BASE_MODEL_KEY
    assert metadata["batchnorm_frozen"] is True
    assert metadata["config"]["source_model_key"] == lf.BASE_MODEL_KEY


def test_lambda_handler_passes_flags_through():
    """lambda_handler 가 from_base/include_processed 를 파이프라인과 응답에 전달"""
    captured = {}

    def fake_pipeline(event):
        captured["event"] = event
        return {
            "success": True,
            "new_version": "v6_feedback_test",
            "source_model_key": lf.BASE_MODEL_KEY,
            "from_base": True,
            "include_processed": True,
        }

    with patch.object(lf, "get_pending_count", return_value=0), patch.object(
        lf, "run_training_pipeline", side_effect=fake_pipeline
    ):
        response = lf.lambda_handler(
            {
                "trigger_type": "manual",
                "force": True,
                "from_base": True,
                "include_processed": True,
            },
            None,
        )

    assert captured["event"]["from_base"] is True
    body = json.loads(response["body"])
    assert response["statusCode"] == 200
    assert body["from_base"] is True
    assert body["include_processed"] is True
    assert body["source_model_key"] == lf.BASE_MODEL_KEY


def test_no_direct_list_objects_v2_in_trainer():
    """1000개 초과 키 누락 방지: list_objects_v2 직접 호출 금지"""
    with io.open("lambda_trainer/lambda_function.py", encoding="utf-8") as f:
        source = f.read()

    assert "s3.list_objects_v2(" not in source
    assert 'get_paginator("list_objects_v2")' in source


@pytest.mark.parametrize("flag", ["from_base", "include_processed"])
def test_flags_default_false(flag):
    """이벤트에 플래그가 없으면 False 로 동작"""
    patches = _patched_pipeline(200)
    for p in patches:
        p.start()
    try:
        result = lf.run_training_pipeline({"trigger_type": "scheduled"})
    finally:
        for p in patches:
            p.stop()

    assert result[flag] is False
