"""
Trainer Lambda 정기 재학습 모드 / 암묵 피드백 샘플 회귀 테스트

대상: lambda_trainer/lambda_function.py

검증 항목:
1. trigger_type=scheduled 이면 from_base / include_processed 기본값 True (event 명시값 우선),
   data_threshold / manual 은 기존과 동일하게 False
2. 정기 실행의 MIN_SAMPLES 는 "새 pending 샘플 수(중복 제거 후)" 로 판정
3. 정기(from_base) 실행의 게이트 기준선은 models/current 모델
4. feedback/pending/implicit/ 하위 prefix 로드 + processed/implicit/ 로 경로 구조 유지 이동
5. sample_weight 로드 기본값(없으면 1.0), 가중 MSE, 평균 1 정규화
6. (analysis_id, 스타일) 중복 제거 우선순위: 명시 > share > save > synthesize >
   shown_not_chosen, 동순위면 최신 timestamp
7. 게이트 평가는 가중치 없이 계산, 리포트/config 에 명시·암묵 샘플 수 기록

AWS 호출은 전부 목(mock) 처리한다.
"""

import importlib.util
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
    """paginator(prefix 하위 전체 나열) / get / put / copy / delete 를 기록하는 목 S3"""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.exceptions = _S3Exceptions
        self.put_objects = {}
        self.copied = []
        self.deleted = []

    def get_paginator(self, operation_name):
        assert operation_name == "list_objects_v2"
        outer = self

        class _Paginator:
            def paginate(self, Bucket=None, Prefix=None, **kwargs):
                # 실제 S3 처럼 Delimiter 없이 하위 prefix 까지 전부 나열
                keys = sorted(k for k in outer.objects if k.startswith(Prefix))
                yield {"Contents": [{"Key": k} for k in keys]}

        return _Paginator()

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

    def copy_object(self, Bucket=None, CopySource=None, Key=None, **kwargs):
        self.copied.append((CopySource["Key"], Key))
        return {}

    def delete_object(self, Bucket=None, Key=None, **kwargs):
        self.deleted.append(Key)
        return {}


def _npz_bytes(gt, metadata=None, sample_weight=None):
    """save_feedback 과 같은 레이아웃의 NPZ 1건 (+ 선택적 sample_weight)"""
    arrays = {
        "face_features": np.array(
            [1.2, 100.0, 130.0, 105.0, 0.78, 0.80], dtype=np.float32
        ),
        "skin_features": np.array([70.0, 15.0], dtype=np.float32),
        "style_embedding": np.zeros(384, dtype=np.float32),
        "ground_truth": np.array([gt], dtype=np.float32),
    }
    if metadata is not None:
        arrays["metadata"] = np.array([json.dumps(metadata)], dtype=str)
    if sample_weight is not None:
        arrays["sample_weight"] = np.array([sample_weight], dtype=np.float32)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _explicit(analysis_id, hairstyle_id, feedback="good", ts="2026-10-01T00:00:00"):
    return _npz_bytes(
        90.0 if feedback == "good" else 10.0,
        {
            "analysis_id": analysis_id,
            "hairstyle_id": hairstyle_id,
            "feedback": feedback,
            "timestamp": ts + "+00:00",
        },
    )


IMPLICIT_GT = {
    "share": 85.0,
    "save": 80.0,
    "synthesize": 55.0,
    "shown_not_chosen": 40.0,
}


def _implicit(analysis_id, hairstyle_id, signal, weight=0.5, ts="2026-10-01T00:00:00"):
    return _npz_bytes(
        IMPLICIT_GT[signal],
        {
            "analysis_id": analysis_id,
            "hairstyle_id": hairstyle_id,
            "source": "implicit",
            "signal_type": signal,
            "rec_rank": 1,
            "ml_rank": 2,
            "propensity": 0.4,
            "timestamp": ts + "+00:00",
        },
        sample_weight=weight,
    )


def _load(fake_s3, include_processed=False):
    with patch.object(lf, "get_s3_client", return_value=fake_s3):
        return lf.load_pending_feedbacks(include_processed=include_processed)


# ========== (1) 트리거별 플래그 기본값 ==========


def test_scheduled_defaults_to_full_retrain_from_base():
    flags = lf.resolve_training_flags(
        {"trigger_type": "scheduled", "schedule": "weekly"}
    )
    assert flags["from_base"] is True
    assert flags["include_processed"] is True
    assert flags["scheduled"] is True


def test_scheduled_explicit_event_values_win():
    flags = lf.resolve_training_flags(
        {"trigger_type": "scheduled", "from_base": False, "include_processed": False}
    )
    assert flags["from_base"] is False
    assert flags["include_processed"] is False


@pytest.mark.parametrize("trigger_type", ["data_threshold", "manual", None])
def test_non_scheduled_defaults_unchanged(trigger_type):
    event = {} if trigger_type is None else {"trigger_type": trigger_type}
    flags = lf.resolve_training_flags(event)
    assert flags["from_base"] is False
    assert flags["include_processed"] is False


def test_min_samples_default_is_30(monkeypatch):
    """MLOPS_MIN_SAMPLES 가 없으면 30 (있으면 env 우선)"""
    path = "lambda_trainer/lambda_function.py"

    def _fresh(name):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    monkeypatch.delenv("MLOPS_MIN_SAMPLES", raising=False)
    assert _fresh("lf_min_default").MIN_SAMPLES == 30

    monkeypatch.setenv("MLOPS_MIN_SAMPLES", "50")
    assert _fresh("lf_min_env").MIN_SAMPLES == 50


def test_lambda_handler_reports_scheduled_defaults():
    captured = {}

    def fake_pipeline(event):
        captured["event"] = event
        return {"success": False, "message": "x", "model_promoted": False}

    with patch.object(lf, "get_pending_count", return_value=10_000), patch.object(
        lf, "run_training_pipeline", side_effect=fake_pipeline
    ):
        response = lf.lambda_handler(
            {"trigger_type": "scheduled", "schedule": "weekly"}, None
        )

    body = json.loads(response["body"])
    assert body["from_base"] is True
    assert body["include_processed"] is True
    # 이벤트 원본을 그대로 넘기고, 기본값 해석은 파이프라인이 같은 함수로 한다
    assert captured["event"] == {"trigger_type": "scheduled", "schedule": "weekly"}


# ========== (2)/(3) 정기 실행 파이프라인 ==========


def _pipeline_data(pending, processed):
    """(pending 샘플 수, processed 샘플 수) 로 로더 반환값 생성"""
    total = pending + processed
    face = np.tile(
        np.array([1.2, 100.0, 130.0, 105.0, 0.78, 0.80], dtype=np.float32), (total, 1)
    )
    skin = np.tile(np.array([70.0, 15.0], dtype=np.float32), (total, 1))
    style = np.zeros((total, 384), dtype=np.float32)
    gt = np.array([90.0 if i % 2 == 0 else 10.0 for i in range(total)], np.float32)
    keys = [f"feedback/pending/p{i}.npz" for i in range(pending)] + [
        f"feedback/processed/batch_a/q{i}.npz" for i in range(processed)
    ]
    analysis_ids = [f"analysis-{i}" for i in range(total)]
    return face, skin, style, gt, total, keys, analysis_ids, None


def _patch_pipeline(data, start_model, current_model, eval_side_effect=None):
    def _load_model(allow_random_init=False, from_base=False):
        return (start_model if from_base else current_model), {"version": "v6"}

    return [
        patch.object(lf, "load_pending_feedbacks", return_value=data),
        patch.object(lf, "load_base_model", side_effect=_load_model),
        patch.object(
            lf,
            "evaluate_holdout",
            side_effect=eval_side_effect
            or (lambda *a: {"mse": 0.1, "ranking_accuracy": 0.8, "num_pairs": 10}),
        ),
        patch.object(
            lf, "fine_tune_model", return_value=(start_model, {"final_loss": 0.1})
        ),
        patch.object(lf, "save_model_to_s3", return_value=True),
        patch.object(lf, "save_rejected_model", return_value=True),
        patch.object(lf, "move_pending_to_processed", return_value=True),
        patch.object(lf, "update_metadata", return_value=None),
        patch.object(lf, "save_evaluation_report", return_value=True),
    ]


def _run(patches, event):
    for p in patches:
        p.start()
    try:
        result = lf.run_training_pipeline(event)
        calls = {
            "load": lf.load_pending_feedbacks.call_args,
            "eval": list(lf.evaluate_holdout.call_args_list),
            "fine_tune": lf.fine_tune_model.call_args,
            "move": lf.move_pending_to_processed.call_args,
            "report": lf.save_evaluation_report.call_args,
        }
    finally:
        for p in patches:
            p.stop()
    return result, calls


def test_scheduled_pipeline_trains_from_base_on_all_data():
    start, current = MagicMock(name="base"), MagicMock(name="current")
    patches = _patch_pipeline(_pipeline_data(40, 800), start, current)
    result, calls = _run(patches, {"trigger_type": "scheduled", "schedule": "weekly"})

    assert result["success"] is True
    assert result["from_base"] is True
    assert result["include_processed"] is True
    assert result["source_model_key"] == lf.BASE_MODEL_KEY
    assert calls["load"].kwargs["include_processed"] is True
    assert result["samples_trained"] == 840
    assert result["new_pending_count"] == 40
    # 홀드아웃이 충분히 확보된다 (840 * 0.15 근사)
    assert result["holdout_size"] >= lf.MIN_HOLDOUT_SAMPLES
    # pending 만 이동
    moved = calls["move"].args[0]
    assert len(moved) == 40
    assert all(k.startswith(lf.PENDING_PREFIX) for k in moved)


def test_scheduled_gate_baseline_is_current_model():
    """from_base 재학습이어도 게이트는 지금 서빙 중인 current 모델과 비교한다"""
    start, current = MagicMock(name="base"), MagicMock(name="current")
    patches = _patch_pipeline(_pipeline_data(40, 300), start, current)
    result, calls = _run(patches, {"trigger_type": "scheduled"})

    before_call, after_call = calls["eval"]
    assert before_call.args[0] is current
    assert after_call.args[0] is start  # fine_tune_model 이 돌려준 학습 후 모델
    assert result["baseline_model_key"] == lf.CURRENT_MODEL_KEY
    assert result["gate"]["baseline_model_key"] == lf.CURRENT_MODEL_KEY


def test_scheduled_gate_baseline_falls_back_to_base_when_current_missing():
    start = MagicMock(name="base")
    patches = _patch_pipeline(_pipeline_data(40, 300), start, None)
    result, calls = _run(patches, {"trigger_type": "scheduled"})

    assert calls["eval"][0].args[0] is start
    assert result["baseline_model_key"] == lf.BASE_MODEL_KEY


def test_scheduled_skips_when_new_pending_below_min_samples():
    """processed 가 많아도 새 pending 이 MIN_SAMPLES 미만이면 학습하지 않는다"""
    start, current = MagicMock(), MagicMock()
    pending = lf.MIN_SAMPLES - 1
    patches = _patch_pipeline(_pipeline_data(pending, 800), start, current)
    result, calls = _run(patches, {"trigger_type": "scheduled"})

    assert result["success"] is False
    assert "Insufficient data" in result["message"]
    assert "new pending" in result["message"]
    assert result["new_pending_count"] == pending
    assert calls["fine_tune"] is None


def test_scheduled_force_bypasses_new_pending_check():
    start, current = MagicMock(), MagicMock()
    patches = _patch_pipeline(_pipeline_data(1, 300), start, current)
    result, calls = _run(patches, {"trigger_type": "scheduled", "force": True})

    assert calls["fine_tune"] is not None
    assert result["success"] is True


def test_manual_include_processed_keeps_total_count_check():
    """manual / data_threshold 의 MIN_SAMPLES 판정은 기존과 같이 로드된 전체 샘플 수"""
    start, current = MagicMock(), MagicMock()
    patches = _patch_pipeline(_pipeline_data(1, 300), start, current)
    result, calls = _run(patches, {"trigger_type": "manual", "include_processed": True})

    assert calls["fine_tune"] is not None
    # from_base 가 아니면 기준선은 시작점(current) 모델 그대로
    assert calls["eval"][0].args[0] is current


# ========== (4) 하위 prefix 로드 / 이동 ==========


def test_loader_reads_implicit_subprefix_with_weights():
    fake_s3 = FakeS3(
        {
            "feedback/pending/2026-10-01_aaaaaaaa_s3_abc123.npz": _explicit("A", 3),
            "feedback/pending/implicit/B_7.npz": _implicit("B", 7, "share", weight=0.4),
        }
    )
    face, skin, style, gt, count, keys, ids, info = _load(fake_s3)

    assert count == 2
    order = info["sample_keys"]
    assert order == sorted(fake_s3.objects)
    by_key = dict(
        zip(
            order,
            zip(info["sources"], info["signal_types"], info["sample_weights"], ids, gt),
        )
    )

    src, sig, w, aid, label = by_key["feedback/pending/implicit/B_7.npz"]
    assert (src, sig, aid, label) == ("implicit", "share", "B", 85.0)
    assert w == pytest.approx(0.4)

    src, sig, w, aid, label = by_key[
        "feedback/pending/2026-10-01_aaaaaaaa_s3_abc123.npz"
    ]
    # 명시 피드백 npz 에는 sample_weight 가 없다 -> 1.0
    assert (src, sig, aid, label) == ("explicit", None, "A", 90.0)
    assert w == 1.0


def test_loader_implicit_fallback_ids_from_filename():
    """metadata 가 없는 암묵 npz 도 파일명에서 analysis_id / hairstyle_id 를 복원"""
    analysis_id = "0f3c9a1e-1111-2222-3333-444455556666"
    key = f"feedback/pending/implicit/{analysis_id}_12.npz"
    assert lf.analysis_id_from_key(key) == analysis_id
    assert lf.extract_style_key({}, key) == "id:12"
    assert lf.is_implicit_key(key)
    assert lf.is_implicit_key("feedback/processed/implicit/batch_x/a_1.npz")
    assert not lf.is_implicit_key("feedback/pending/2026-10-01_aaaaaaaa_s3_x.npz")
    # 기존 명시 피드백 파일명 규칙은 그대로
    assert (
        lf.analysis_id_from_key("feedback/pending/2026-10-01_aaaaaaaa_s3_x.npz")
        == "aaaaaaaa"
    )


def test_loader_include_processed_reads_processed_implicit():
    fake_s3 = FakeS3(
        {
            "feedback/pending/implicit/A_1.npz": _implicit("A", 1, "save"),
            "feedback/processed/implicit/batch_x/B_2.npz": _implicit("B", 2, "share"),
            "feedback/processed/batch_x/2026-09-01_cccccccc_s5_aaaaaa.npz": _explicit(
                "C", 5
            ),
        }
    )
    *_, count, keys, ids, info = _load(fake_s3, include_processed=True)
    assert count == 3
    assert set(ids) == {"A", "B", "C"}

    *_, count, keys, ids, info = _load(fake_s3, include_processed=False)
    assert count == 1
    assert ids == ["A"]


def test_processed_key_preserves_subprefix():
    assert (
        lf.processed_key_for("feedback/pending/implicit/A_1.npz", "batch_1")
        == "feedback/processed/implicit/batch_1/A_1.npz"
    )
    assert (
        lf.processed_key_for("feedback/pending/x.npz", "batch_1")
        == "feedback/processed/batch_1/x.npz"
    )


def test_move_pending_to_processed_keeps_implicit_structure():
    fake_s3 = FakeS3()
    keys = [
        "feedback/pending/x.npz",
        "feedback/pending/implicit/A_1.npz",
        "feedback/processed/implicit/batch_old/B_2.npz",
    ]
    with patch.object(lf, "get_s3_client", return_value=fake_s3):
        assert lf.move_pending_to_processed(keys, "batch_t") is True

    assert fake_s3.copied == [
        ("feedback/pending/x.npz", "feedback/processed/batch_t/x.npz"),
        (
            "feedback/pending/implicit/A_1.npz",
            "feedback/processed/implicit/batch_t/A_1.npz",
        ),
    ]
    assert fake_s3.deleted == [
        "feedback/pending/x.npz",
        "feedback/pending/implicit/A_1.npz",
    ]


# ========== (5) 가중치 ==========


def test_loader_rejects_invalid_weight_sample():
    fake_s3 = FakeS3(
        {
            "feedback/pending/implicit/A_1.npz": _implicit(
                "A", 1, "save", weight=float("nan")
            ),
            "feedback/pending/implicit/B_1.npz": _implicit("B", 1, "save", weight=0.3),
        }
    )
    *_, count, keys, ids, info = _load(fake_s3)
    assert ids == ["B"]


def test_normalize_sample_weights_mean_one():
    w = lf.normalize_sample_weights(np.array([1.0, 0.5, 0.5, 2.0]), 4)
    assert w.mean() == pytest.approx(1.0)
    np.testing.assert_allclose(w / w[0], [1.0, 0.5, 0.5, 2.0], rtol=1e-6)
    np.testing.assert_array_equal(lf.normalize_sample_weights(None, 3), np.ones(3))
    # 비정상 입력은 전부 1.0
    np.testing.assert_array_equal(
        lf.normalize_sample_weights(np.array([0.0, 0.0]), 2), np.ones(2)
    )
    np.testing.assert_array_equal(
        lf.normalize_sample_weights(np.array([1.0, -1.0]), 2), np.ones(2)
    )
    np.testing.assert_array_equal(
        lf.normalize_sample_weights(np.array([1.0]), 2), np.ones(2)
    )


def test_weighted_mse_equals_mse_when_uniform():
    pred = torch.tensor([0.1, 0.5, 0.9])
    target = torch.tensor([0.0, 1.0, 0.5])
    ones = torch.ones(3)
    assert lf.weighted_mse_loss(pred, target, ones).item() == pytest.approx(
        torch.nn.functional.mse_loss(pred, target).item()
    )
    w = torch.tensor([2.0, 0.0, 1.0])
    expected = (2.0 * 0.01 + 0.0 + 1.0 * 0.16) / 3
    assert lf.weighted_mse_loss(pred, target, w).item() == pytest.approx(expected)


def test_feedback_dataset_exposes_normalized_weights():
    face = np.tile(np.array([1.2, 100.0, 130.0, 105.0, 0.78, 0.80], np.float32), (2, 1))
    skin = np.tile(np.array([70.0, 15.0], np.float32), (2, 1))
    style = np.zeros((2, 384), np.float32)
    gt = np.array([90.0, 40.0], np.float32)

    ds = lf.FeedbackDataset(face, skin, style, gt, sample_weights=np.array([1.0, 0.5]))
    assert ds.sample_weights.mean().item() == pytest.approx(1.0)
    assert len(ds[0]) == 5

    ds_default = lf.FeedbackDataset(face, skin, style, gt)
    assert ds_default.sample_weights.tolist() == [1.0, 1.0]


def test_fine_tune_uses_sample_weights():
    """가중치 0 인 샘플은 학습에 영향을 주지 않는다 (가중 손실이 실제로 쓰인다)"""
    torch.manual_seed(0)
    base = lf.RecommendationModelV6()
    n = 8
    rng = np.random.default_rng(0)
    face = np.tile(np.array([1.2, 450.0, 560.0, 450.0, 0.82, 0.80], np.float32), (n, 1))
    face += rng.normal(0, 1, face.shape).astype(np.float32)
    skin = np.tile(np.array([80.0, 12.0], np.float32), (n, 1))
    style = rng.normal(0, 1, (n, 384)).astype(np.float32)
    gt = np.array([90.0, 10.0] * (n // 2), np.float32)

    def _train(weights, gt_values):
        model = lf.RecommendationModelV6()
        model.load_state_dict(base.state_dict())
        torch.manual_seed(1)
        trained, _ = lf.fine_tune_model(
            model, face, skin, style, gt_values, epochs=2, sample_weights=weights
        )
        return trained.fc_out.weight.detach().clone()

    weights = np.array([1.0, 0.0] * (n // 2), np.float32)
    gt_changed = gt.copy()
    gt_changed[1::2] = 95.0  # 가중치 0 샘플의 라벨만 바꾼다
    torch.manual_seed(1)
    a = _train(weights, gt)
    b = _train(weights, gt_changed)
    torch.testing.assert_close(a, b)


# ========== (6) 중복 제거 ==========


def test_dedupe_priority_explicit_over_all_implicit():
    fake_s3 = FakeS3(
        {
            "feedback/pending/implicit/A_1.npz": _implicit(
                "A", 1, "share", ts="2026-10-05T00:00:00"
            ),
            "feedback/processed/batch_x/2026-09-01_A_s1_aaaaaa.npz": _explicit(
                "A", 1, "bad", ts="2026-09-01T00:00:00"
            ),
        }
    )
    *_, gt, count, keys, ids, info = _load(fake_s3, include_processed=True)
    assert count == 1
    assert info["sources"] == ["explicit"]
    assert gt.tolist() == [10.0]
    assert info["duplicates_dropped"] == 1
    # 빠진 pending 파일도 file_keys 에 남아 성공 시 함께 이동된다
    assert "feedback/pending/implicit/A_1.npz" in keys


@pytest.mark.parametrize(
    "signals,winner",
    [
        (["shown_not_chosen", "synthesize", "save", "share"], "share"),
        (["shown_not_chosen", "synthesize", "save"], "save"),
        (["shown_not_chosen", "synthesize"], "synthesize"),
    ],
)
def test_dedupe_implicit_signal_priority(signals, winner):
    records = [
        {
            "key": f"k{i}",
            "analysis_id": "A",
            "style_key": "id:1",
            "source": "implicit",
            "signal_type": s,
            "priority": lf.sample_priority("implicit", s),
            # 우선순위가 낮은 신호일수록 더 최신이어도 진다
            "timestamp": lf.parse_timestamp(f"2026-10-0{i + 1}T00:00:00+00:00"),
        }
        for i, s in enumerate(reversed(signals))
    ]
    kept, dropped = lf.dedupe_feedback_records(records)
    assert dropped == len(signals) - 1
    assert [r["signal_type"] for r in kept] == [winner]


def test_dedupe_same_priority_keeps_latest_timestamp():
    fake_s3 = FakeS3(
        {
            "feedback/pending/2026-10-01_A_s1_old.npz": _explicit(
                "A", 1, "good", ts="2026-10-01T00:00:00"
            ),
            "feedback/pending/2026-10-02_A_s1_new.npz": _explicit(
                "A", 1, "bad", ts="2026-10-02T00:00:00"
            ),
        }
    )
    *_, gt, count, keys, ids, info = _load(fake_s3)
    assert count == 1
    assert gt.tolist() == [10.0]
    assert info["sample_keys"] == ["feedback/pending/2026-10-02_A_s1_new.npz"]


def test_dedupe_keeps_different_styles_and_unidentified_samples():
    fake_s3 = FakeS3(
        {
            "feedback/pending/implicit/A_1.npz": _implicit("A", 1, "save"),
            "feedback/pending/implicit/A_2.npz": _implicit("A", 2, "save"),
            "feedback/pending/implicit/B_1.npz": _implicit("B", 1, "save"),
            # metadata 없는 구형 npz 는 스타일을 알 수 없어 중복 제거하지 않는다
            "feedback/pending/legacy1.npz": _npz_bytes(90.0),
            "feedback/pending/legacy2.npz": _npz_bytes(90.0),
        }
    )
    *_, count, keys, ids, info = _load(fake_s3)
    assert count == 5
    assert info["duplicates_dropped"] == 0


def test_dedupe_trending_style_name():
    meta = {"analysis_id": "A", "style_name": "울프컷", "source": "trending"}
    assert lf.extract_style_key(meta, "feedback/pending/x.npz") == "name:울프컷"
    # trending 은 명시 피드백으로 취급
    assert lf.classify_sample(meta, "feedback/pending/x.npz") == ("explicit", None)


# ========== (7) 게이트 무가중 + 리포트 ==========


def _weighted_loader_data():
    """명시 120 + 암묵 120 (가중치 0.25) - pending"""
    n = 240
    face = np.tile(np.array([1.2, 100.0, 130.0, 105.0, 0.78, 0.80], np.float32), (n, 1))
    skin = np.tile(np.array([70.0, 15.0], np.float32), (n, 1))
    style = np.zeros((n, 384), np.float32)
    signals = ["share", "save", "synthesize", "shown_not_chosen"]
    sources, signal_types, weights, gts, keys = [], [], [], [], []
    for i in range(n):
        if i % 2 == 0:
            sources.append("explicit")
            signal_types.append(None)
            weights.append(1.0)
            gts.append(90.0 if i % 4 == 0 else 10.0)
            keys.append(f"feedback/pending/e{i}.npz")
        else:
            s = signals[(i // 2) % 4]
            sources.append("implicit")
            signal_types.append(s)
            weights.append(0.25)
            gts.append(IMPLICIT_GT[s])
            keys.append(f"feedback/pending/implicit/a{i}_{i}.npz")
    analysis_ids = [f"analysis-{i // 2}" for i in range(n)]
    info = {
        "sample_keys": keys,
        "sample_weights": np.array(weights, np.float32),
        "sources": sources,
        "signal_types": signal_types,
        "duplicates_dropped": 3,
    }
    return (face, skin, style, np.array(gts, np.float32), n, keys, analysis_ids, info)


def test_pipeline_passes_train_weights_and_evaluates_unweighted():
    data = _weighted_loader_data()
    start, current = MagicMock(), MagicMock()
    patches = _patch_pipeline(data, start, current)
    result, calls = _run(patches, {"trigger_type": "manual"})

    assert result["success"] is True
    train_idx, holdout_idx = lf.split_holdout_indices(data[6], lf.TRAIN_HOLDOUT_RATIO)

    # fine-tune 에는 학습 분할의 가중치만
    passed = calls["fine_tune"].kwargs["sample_weights"]
    np.testing.assert_allclose(passed, data[7]["sample_weights"][train_idx])

    # 게이트 평가는 가중치 인자 없이 (model, face, skin, style, gt, ids)
    for call in calls["eval"]:
        assert len(call.args) == 6
        assert not call.kwargs


def test_evaluate_holdout_is_unweighted_mse():
    """evaluate_holdout 의 MSE 는 단순 평균 (샘플 가중치와 무관)"""
    model = lf.RecommendationModelV6()
    n = 4
    face = np.tile(np.array([1.2, 450.0, 560.0, 450.0, 0.82, 0.80], np.float32), (n, 1))
    skin = np.tile(np.array([80.0, 12.0], np.float32), (n, 1))
    style = np.zeros((n, 384), np.float32)
    gt = np.array([0.1, 0.9, 0.5, 0.3])
    preds = lf.predict_normalized(model, face, skin, style)
    metrics = lf.evaluate_holdout(model, face, skin, style, gt, ["a", "a", "b", "b"])
    assert metrics["mse"] == pytest.approx(float(np.mean((preds - gt) ** 2)), rel=1e-6)


def test_pipeline_records_sample_composition_in_report_and_config():
    data = _weighted_loader_data()
    start, current = MagicMock(), MagicMock()
    saved = {}

    def _save(model, config, version):
        saved["config"] = dict(config)
        return True

    patches = _patch_pipeline(data, start, current)
    patches[4] = patch.object(lf, "save_model_to_s3", side_effect=_save)
    result, calls = _run(patches, {"trigger_type": "manual"})

    summary = calls["report"].kwargs["data_summary"]
    assert summary["explicit"] == 120
    assert summary["implicit"] == 120
    assert summary["implicit_by_signal_type"] == {
        "save": 30,
        "share": 30,
        "shown_not_chosen": 30,
        "synthesize": 30,
    }
    assert summary["duplicates_dropped"] == 3
    assert summary["train"]["explicit"] + summary["holdout"]["explicit"] == 120
    assert summary["train"]["implicit"] + summary["holdout"]["implicit"] == 120
    assert result["data_summary"] == summary

    config = saved["config"]
    assert config["explicit_samples"] == 120
    assert config["implicit_samples"] == 120
    assert config["implicit_by_signal_type"]["share"] == 30


def test_evaluation_report_contains_data_summary():
    fake_s3 = FakeS3()
    summary = {"explicit": 5, "implicit": 2, "implicit_by_signal_type": {"save": 2}}
    with patch.object(lf, "get_s3_client", return_value=fake_s3):
        assert lf.save_evaluation_report(
            {"mse": 0.1},
            {"mse": 0.09},
            {},
            "v6_test",
            gate={"passed": True},
            data_summary=summary,
        )
    report = json.loads(fake_s3.put_objects["evaluations/v6_test_report.json"])
    assert report["data_summary"] == summary


def test_current_metadata_config_records_sample_counts():
    fake_s3 = FakeS3()
    config = {"version": "v", "explicit_samples": 800, "implicit_samples": 40}
    with patch.object(lf, "get_s3_client", return_value=fake_s3):
        assert lf.save_current_model_metadata(b"bytes", config, "v")
    metadata = json.loads(fake_s3.put_objects["models/current/metadata.json"])
    assert metadata["config"]["explicit_samples"] == 800
    assert metadata["config"]["implicit_samples"] == 40


def test_pipeline_success_moves_deduped_pending_files_too():
    """중복 제거로 학습에서 빠진 pending 파일도 성공 시 processed 로 이동"""
    data = list(_pipeline_data(40, 300))
    data[7] = {"sample_keys": list(data[5]), "duplicates_dropped": 1}
    data[5] = data[5] + ["feedback/pending/implicit/dup_1.npz"]  # 빠진 중복 파일
    start, current = MagicMock(), MagicMock()
    patches = _patch_pipeline(tuple(data), start, current)
    result, calls = _run(patches, {"trigger_type": "scheduled"})

    moved = calls["move"].args[0]
    assert "feedback/pending/implicit/dup_1.npz" in moved
    assert len(moved) == 41
    # 새 pending 수는 학습에 실제로 쓰인 샘플 기준
    assert result["new_pending_count"] == 40


def test_lambda_handler_scheduled_ignores_metadata_counter():
    """정기 실행은 metadata 카운터 대신 파이프라인의 실제 pending 수로 판정한다"""
    with patch.object(lf, "get_pending_count", return_value=0), patch.object(
        lf,
        "run_training_pipeline",
        return_value={
            "success": False,
            "message": "Insufficient data: 3/30 new pending samples",
        },
    ) as pipeline:
        response = lf.lambda_handler({"trigger_type": "scheduled"}, None)

    assert pipeline.called
    assert "new pending" in json.loads(response["body"])["message"]


@pytest.mark.parametrize("trigger_type", ["data_threshold", "manual"])
def test_lambda_handler_non_scheduled_keeps_counter_precheck(trigger_type):
    with patch.object(lf, "get_pending_count", return_value=0), patch.object(
        lf, "run_training_pipeline"
    ) as pipeline:
        response = lf.lambda_handler({"trigger_type": trigger_type}, None)

    assert not pipeline.called
    assert json.loads(response["body"])["min_samples"] == lf.MIN_SAMPLES
