"""
Trainer Lambda 재학습 품질 게이트 회귀 테스트

대상: lambda_trainer/lambda_function.py

검증 항목:
1. analysis_id 해시 기반 홀드아웃 분할의 결정성 / 그룹 보존 / 비율
2. pairwise ranking accuracy (같은 analysis 안에서만, 쌍이 없으면 None)
3. 게이트 판정: MSE 허용치, ranking 허용치, skip_gate, 홀드아웃 부족
4. 통과 시 models/current/model.pt 교체 + pending 이동
5. 거부 시 models/rejected/{version}.pt 에만 저장, current 미교체,
   pending 미이동, pending_count 미리셋 + training_triggered_at 갱신
6. force=True 는 MIN_SAMPLES 만 우회하고 게이트는 우회하지 않음

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
    """put/get/copy/delete 를 기록만 하는 최소 목 S3"""

    def __init__(self, objects=None):
        self.objects = objects or {}
        self.exceptions = _S3Exceptions
        self.put_objects = {}
        self.copied = []
        self.deleted = []

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


# ========== (1) 홀드아웃 분할 ==========


def test_split_is_deterministic_across_calls_and_order():
    """같은 analysis_id 집합이면 순서·호출 시점과 무관하게 같은 분할"""
    ids = [f"analysis-{i}" for i in range(300)]

    train_a, hold_a = lf.split_holdout_indices(ids, 0.15)
    train_b, hold_b = lf.split_holdout_indices(ids, 0.15)

    assert train_a == train_b
    assert hold_a == hold_b

    # 순서를 뒤집어도 "어떤 id 가 홀드아웃인가" 는 동일해야 한다
    reversed_ids = list(reversed(ids))
    _, hold_rev = lf.split_holdout_indices(reversed_ids, 0.15)
    assert {ids[i] for i in hold_a} == {reversed_ids[i] for i in hold_rev}


def test_split_keeps_same_analysis_together():
    """같은 analysis_id 의 샘플은 절대 학습/홀드아웃으로 쪼개지지 않는다"""
    # analysis 당 3개 샘플(스타일 3종 피드백)
    ids = [f"analysis-{i // 3}" for i in range(300)]

    train_idx, hold_idx = lf.split_holdout_indices(ids, 0.15)

    train_ids = {ids[i] for i in train_idx}
    hold_ids = {ids[i] for i in hold_idx}

    assert train_ids & hold_ids == set()
    assert len(train_idx) + len(hold_idx) == len(ids)


def test_split_ratio_is_close_to_target():
    """충분히 많은 id 에서 홀드아웃 비율이 목표치에 근접"""
    ids = [f"analysis-{i}" for i in range(4000)]

    for ratio in (0.1, 0.15, 0.3):
        _, hold_idx = lf.split_holdout_indices(ids, ratio)
        observed = len(hold_idx) / len(ids)
        assert abs(observed - ratio) < 0.02, (ratio, observed)


def test_split_disjoint_and_exhaustive():
    """분할은 겹치지 않고 빠짐없다"""
    ids = [f"analysis-{i}" for i in range(500)]
    train_idx, hold_idx = lf.split_holdout_indices(ids, 0.15)

    assert set(train_idx) & set(hold_idx) == set()
    assert sorted(train_idx + hold_idx) == list(range(500))


def test_split_ratio_zero_keeps_everything_for_training():
    """ratio<=0 이면 홀드아웃 없음 (전부 학습)"""
    ids = [f"analysis-{i}" for i in range(50)]
    train_idx, hold_idx = lf.split_holdout_indices(ids, 0.0)

    assert hold_idx == []
    assert train_idx == list(range(50))


@pytest.mark.parametrize(
    "key,expected",
    [
        ("feedback/pending/2025-12-02_abcd1234_s12_ab12cd.npz", "abcd1234"),
        ("feedback/pending/2025-12-02_trending_ff00aa11_s3_9c8b7a.npz", "ff00aa11"),
        ("feedback/processed/batch_a/2026-01-01_deadbeef_s7_112233.npz", "deadbeef"),
    ],
)
def test_analysis_id_from_key(key, expected):
    """파일명 규칙에서 analysis_id 접두사를 복원한다"""
    assert lf.analysis_id_from_key(key) == expected


def test_extract_analysis_id_prefers_npz_metadata():
    """NPZ metadata 의 analysis_id 가 파일명보다 우선한다 (pickle 없이 읽힘)"""
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        ground_truth=np.array([90.0], dtype=np.float32),
        metadata=np.array(
            [json.dumps({"analysis_id": "full-uuid-1234-5678", "feedback": "good"})],
            dtype=str,
        ),
    )
    buffer.seek(0)
    data = np.load(buffer, allow_pickle=False)

    extracted = lf.extract_analysis_id(
        data, "feedback/pending/2025-12-02_abcd1234_s1_aaaaaa.npz"
    )
    assert extracted == "full-uuid-1234-5678"


def test_extract_analysis_id_falls_back_to_key():
    """metadata 가 없는 레거시 NPZ 는 파일명에서 추출"""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, ground_truth=np.array([10.0], dtype=np.float32))
    buffer.seek(0)
    data = np.load(buffer, allow_pickle=False)

    extracted = lf.extract_analysis_id(
        data, "feedback/pending/2025-12-02_abcd1234_s1_aaaaaa.npz"
    )
    assert extracted == "abcd1234"


# ========== (2) pairwise ranking accuracy ==========


def test_pairwise_ranking_accuracy_perfect_ordering():
    """같은 analysis 안에서 good 이 bad 보다 높으면 1.0"""
    preds = np.array([0.9, 0.2, 0.8, 0.1])
    gts = np.array([1.0, 0.0, 1.0, 0.0])
    ids = ["a", "a", "b", "b"]

    accuracy, pairs = lf.pairwise_ranking_accuracy(preds, gts, ids)

    assert accuracy == 1.0
    assert pairs == 2


def test_pairwise_ranking_accuracy_inverted_ordering():
    """순서가 완전히 뒤집히면 0.0"""
    preds = np.array([0.1, 0.9])
    gts = np.array([1.0, 0.0])

    accuracy, pairs = lf.pairwise_ranking_accuracy(preds, gts, ["a", "a"])

    assert accuracy == 0.0
    assert pairs == 1


def test_pairwise_ranking_ignores_cross_analysis_pairs():
    """서로 다른 analysis 사이의 쌍은 세지 않는다"""
    # analysis 가 전부 달라 유효한 쌍이 없다
    preds = np.array([0.1, 0.9, 0.5])
    gts = np.array([1.0, 0.0, 1.0])

    accuracy, pairs = lf.pairwise_ranking_accuracy(preds, gts, ["a", "b", "c"])

    assert accuracy is None
    assert pairs == 0


def test_pairwise_ranking_none_when_all_labels_equal():
    """같은 analysis 안이라도 라벨이 같으면 비교할 쌍이 없다"""
    preds = np.array([0.3, 0.7])
    gts = np.array([1.0, 1.0])

    accuracy, pairs = lf.pairwise_ranking_accuracy(preds, gts, ["a", "a"])

    assert accuracy is None
    assert pairs == 0


# ========== (3) 게이트 판정 ==========


def _metrics(mse, rank=0.8, pairs=10):
    return {"mse": mse, "ranking_accuracy": rank, "num_pairs": pairs}


def test_gate_passes_when_both_metrics_improve():
    gate = lf.evaluate_quality_gate(_metrics(0.20, 0.70), _metrics(0.15, 0.80), 50)

    assert gate["passed"] is True
    assert gate["reason"] == "passed"
    assert gate["holdout_size"] == 50


def test_gate_allows_mse_within_tolerance():
    """MSE 가 허용치(기본 2%) 안에서 나빠지는 것은 통과"""
    before = _metrics(0.100, 0.80)
    after = _metrics(0.101, 0.80)  # +1%

    assert lf.evaluate_quality_gate(before, after, 50)["passed"] is True


def test_gate_rejects_mse_beyond_tolerance():
    """MSE 가 허용치를 넘어 나빠지면 거부"""
    before = _metrics(0.100, 0.80)
    after = _metrics(0.130, 0.80)  # +30%

    gate = lf.evaluate_quality_gate(before, after, 50)

    assert gate["passed"] is False
    assert gate["reason"] == "mse_regressed"


def test_gate_rejects_ranking_regression_even_if_mse_improves():
    """MSE 가 좋아져도 순위가 무너지면 거부"""
    before = _metrics(0.20, 0.80)
    after = _metrics(0.10, 0.50)

    gate = lf.evaluate_quality_gate(before, after, 50)

    assert gate["passed"] is False
    assert gate["reason"] == "ranking_regressed"


def test_gate_allows_ranking_within_tolerance():
    before = _metrics(0.20, 0.800)
    after = _metrics(0.19, 0.790)  # -0.01, 허용치 0.02 이내

    assert lf.evaluate_quality_gate(before, after, 50)["passed"] is True


def test_gate_uses_mse_only_when_no_pairs():
    """쌍이 없어 ranking 이 None 이면 MSE 만으로 판정"""
    before = {"mse": 0.20, "ranking_accuracy": None, "num_pairs": 0}
    after = {"mse": 0.10, "ranking_accuracy": None, "num_pairs": 0}

    gate = lf.evaluate_quality_gate(before, after, 50)

    assert gate["passed"] is True
    assert gate["reason"] == "passed_mse_only"


def test_gate_rejects_insufficient_holdout():
    """홀드아웃 20개 미만이면 지표와 무관하게 교체 거부"""
    gate = lf.evaluate_quality_gate(_metrics(0.20), _metrics(0.01), holdout_size=19)

    assert gate["passed"] is False
    assert gate["reason"] == "insufficient_holdout"
    assert gate["holdout_size"] == 19


def test_gate_skip_gate_bypasses_everything():
    """skip_gate 는 홀드아웃 부족/지표 악화를 모두 우회한다"""
    gate = lf.evaluate_quality_gate(
        _metrics(0.10, 0.90), _metrics(0.99, 0.10), holdout_size=0, skip_gate=True
    )

    assert gate["passed"] is True
    assert gate["reason"] == "skipped"


def test_gate_reports_required_body_fields():
    """결과 본문 계약: passed / reason / holdout_size / before / after"""
    before = _metrics(0.2)
    after = _metrics(0.1)
    gate = lf.evaluate_quality_gate(before, after, 42)

    for field in ("passed", "reason", "holdout_size", "before", "after"):
        assert field in gate
    assert gate["before"] is before
    assert gate["after"] is after


# ========== (4)/(5) 파이프라인 S3 동작 ==========


def _pipeline_data(count):
    face = np.tile(
        np.array([1.2, 100.0, 130.0, 105.0, 0.78, 0.80], dtype=np.float32), (count, 1)
    )
    skin = np.tile(np.array([70.0, 15.0], dtype=np.float32), (count, 1))
    style = np.zeros((count, 384), dtype=np.float32)
    gt = np.array([90.0 if i % 2 == 0 else 10.0 for i in range(count)], np.float32)
    keys = [f"feedback/pending/f{i}.npz" for i in range(count)]
    analysis_ids = [f"analysis-{i}" for i in range(count)]
    return face, skin, style, gt, count, keys, analysis_ids


def _run_pipeline(fake_s3, before, after, event=None, count=200):
    """
    save_model_to_s3 / save_rejected_model / move_pending_to_processed /
    update_metadata 는 실제 구현을 쓰고 S3 만 목으로 바꾼다.
    """
    model = lf.RecommendationModelV6()
    metrics = [before, after]

    def eval_spy(m, f, s, st, gt, ids):
        return metrics.pop(0)

    patches = [
        patch.object(lf, "get_s3_client", return_value=fake_s3),
        patch.object(lf, "load_pending_feedbacks", return_value=_pipeline_data(count)),
        patch.object(lf, "load_base_model", return_value=(model, {"version": "v6"})),
        patch.object(lf, "evaluate_holdout", side_effect=eval_spy),
        patch.object(lf, "fine_tune_model", return_value=(model, {"final_loss": 0.12})),
        patch.object(lf, "backup_lambda_config", return_value={}),
        patch.object(lf, "update_analyze_lambda_envvars", return_value=True),
    ]

    for p in patches:
        p.start()
    try:
        return lf.run_training_pipeline(event or {"trigger_type": "scheduled"})
    finally:
        for p in patches:
            p.stop()


def test_pipeline_gate_pass_replaces_current_model_and_moves_pending():
    """통과: models/current/model.pt 교체 + archive 저장 + pending 이동"""
    fake_s3 = FakeS3(
        objects={
            "feedback/metadata.json": json.dumps(
                {"pending_count": 200, "total_feedback_count": 1000}
            )
        }
    )

    result = _run_pipeline(fake_s3, _metrics(0.20, 0.70), _metrics(0.10, 0.85))

    assert result["success"] is True
    assert result["model_promoted"] is True
    assert result["gate"]["passed"] is True

    version = result["new_version"]
    assert "models/current/model.pt" in fake_s3.put_objects
    assert f"models/archive/{version}.pt" in fake_s3.put_objects
    assert "models/current/metadata.json" in fake_s3.put_objects
    # 거부 경로의 키는 생기지 않는다
    assert not any(k.startswith(lf.REJECTED_MODEL_PREFIX) for k in fake_s3.put_objects)

    # pending 파일 이동
    assert result["pending_files_moved"] == 200
    assert len(fake_s3.deleted) == 200
    assert all(k.startswith(lf.PENDING_PREFIX) for k in fake_s3.deleted)

    # 성공 시에는 pending_count 를 리셋한다
    metadata = json.loads(fake_s3.put_objects["feedback/metadata.json"])
    assert metadata["pending_count"] == 0
    assert metadata["last_training_at"]


def test_pipeline_gate_reject_keeps_current_model():
    """거부: rejected/ 에만 저장하고 current/archive 는 건드리지 않는다"""
    fake_s3 = FakeS3(
        objects={
            "feedback/metadata.json": json.dumps(
                {"pending_count": 200, "total_feedback_count": 1000}
            )
        }
    )

    # MSE 가 허용치를 크게 넘어 악화
    result = _run_pipeline(fake_s3, _metrics(0.10, 0.80), _metrics(0.40, 0.80))

    assert result["success"] is False
    assert result["model_promoted"] is False
    assert result["gate"]["passed"] is False
    assert result["gate"]["reason"] == "mse_regressed"

    version = result["new_version"]
    assert f"{lf.REJECTED_MODEL_PREFIX}{version}.pt" in fake_s3.put_objects
    assert "models/current/model.pt" not in fake_s3.put_objects
    assert f"models/archive/{version}.pt" not in fake_s3.put_objects
    assert "models/current/metadata.json" not in fake_s3.put_objects


def test_pipeline_gate_reject_writes_evaluation_report_with_reason():
    """거부 사유가 evaluations/{version}_report.json 에 남는다"""
    fake_s3 = FakeS3(
        objects={"feedback/metadata.json": json.dumps({"pending_count": 200})}
    )

    result = _run_pipeline(fake_s3, _metrics(0.20, 0.90), _metrics(0.10, 0.40))

    version = result["new_version"]
    report_key = f"evaluations/{version}_report.json"
    assert report_key in fake_s3.put_objects

    report = json.loads(fake_s3.put_objects[report_key])
    assert report["gate"]["passed"] is False
    assert report["gate"]["reason"] == "ranking_regressed"
    assert report["model_promoted"] is False
    assert report["gate"]["holdout_size"] == result["holdout_size"]


def test_pipeline_gate_reject_does_not_move_pending():
    """거부 시 pending 파일은 그대로 남아 다음 학습에 재포함된다"""
    fake_s3 = FakeS3(
        objects={"feedback/metadata.json": json.dumps({"pending_count": 200})}
    )

    result = _run_pipeline(fake_s3, _metrics(0.10, 0.80), _metrics(0.40, 0.80))

    assert result["pending_files_moved"] == 0
    assert fake_s3.copied == []
    assert fake_s3.deleted == []


def test_pipeline_gate_reject_keeps_pending_count_and_bumps_trigger_time():
    """거부 시 pending_count 는 유지, training_triggered_at 만 갱신(트리거 폭주 방지)"""
    fake_s3 = FakeS3(
        objects={
            "feedback/metadata.json": json.dumps(
                {"pending_count": 200, "last_training_at": "2026-01-01T00:00:00+00:00"}
            )
        }
    )

    _run_pipeline(fake_s3, _metrics(0.10, 0.80), _metrics(0.40, 0.80))

    metadata = json.loads(fake_s3.put_objects["feedback/metadata.json"])
    assert metadata["pending_count"] == 200
    # 성공 시각은 갱신되지 않는다
    assert metadata["last_training_at"] == "2026-01-01T00:00:00+00:00"
    # 쿨다운 기준 시각만 갱신
    assert metadata["training_triggered_at"]


def test_pipeline_insufficient_holdout_rejects_replacement():
    """홀드아웃이 20개 미만이면 지표가 좋아도 교체하지 않는다"""
    fake_s3 = FakeS3(
        objects={"feedback/metadata.json": json.dumps({"pending_count": 60})}
    )

    # 60 샘플 -> 홀드아웃 9건
    result = _run_pipeline(
        fake_s3, _metrics(0.90, 0.10), _metrics(0.01, 0.99), count=60
    )

    assert result["holdout_size"] < lf.MIN_HOLDOUT_SAMPLES
    assert result["gate"]["reason"] == "insufficient_holdout"
    assert result["model_promoted"] is False
    assert "models/current/model.pt" not in fake_s3.put_objects


def test_pipeline_skip_gate_promotes_despite_regression():
    """skip_gate=True 는 지표가 나빠져도 교체한다 (의도적 전체 재학습용)"""
    fake_s3 = FakeS3(
        objects={"feedback/metadata.json": json.dumps({"pending_count": 200})}
    )

    result = _run_pipeline(
        fake_s3,
        _metrics(0.10, 0.90),
        _metrics(0.99, 0.10),
        event={"trigger_type": "manual", "skip_gate": True},
    )

    assert result["success"] is True
    assert result["model_promoted"] is True
    assert result["gate"]["reason"] == "skipped"
    assert "models/current/model.pt" in fake_s3.put_objects


def test_pipeline_force_does_not_bypass_gate():
    """force=True 는 MIN_SAMPLES 만 우회하고 품질 게이트는 그대로 적용된다"""
    fake_s3 = FakeS3(
        objects={"feedback/metadata.json": json.dumps({"pending_count": 200})}
    )

    result = _run_pipeline(
        fake_s3,
        _metrics(0.10, 0.80),
        _metrics(0.40, 0.80),
        event={"trigger_type": "manual", "force": True},
    )

    assert result["gate"]["passed"] is False
    assert result["model_promoted"] is False
    assert "models/current/model.pt" not in fake_s3.put_objects
    assert f"{lf.REJECTED_MODEL_PREFIX}{result['new_version']}.pt" in (
        fake_s3.put_objects
    )


def test_pipeline_trains_only_on_train_split():
    """홀드아웃 샘플은 fine_tune_model 에 전달되지 않는다"""
    fake_s3 = FakeS3(
        objects={"feedback/metadata.json": json.dumps({"pending_count": 200})}
    )
    model = lf.RecommendationModelV6()
    captured = {}

    def fake_fine_tune(m, face, skin, style, gt, **kwargs):
        captured["train_size"] = len(gt)
        return model, {"final_loss": 0.1}

    metrics = [_metrics(0.2, 0.8), _metrics(0.1, 0.85)]

    patches = [
        patch.object(lf, "get_s3_client", return_value=fake_s3),
        patch.object(lf, "load_pending_feedbacks", return_value=_pipeline_data(200)),
        patch.object(lf, "load_base_model", return_value=(model, {"version": "v6"})),
        patch.object(lf, "evaluate_holdout", side_effect=lambda *a: metrics.pop(0)),
        patch.object(lf, "fine_tune_model", side_effect=fake_fine_tune),
        patch.object(lf, "backup_lambda_config", return_value={}),
        patch.object(lf, "update_analyze_lambda_envvars", return_value=True),
    ]
    for p in patches:
        p.start()
    try:
        result = lf.run_training_pipeline({"trigger_type": "scheduled"})
    finally:
        for p in patches:
            p.stop()

    assert result["holdout_size"] > 0
    assert captured["train_size"] == result["train_size"]
    assert result["train_size"] + result["holdout_size"] == 200
    assert captured["train_size"] < 200


def test_pipeline_body_contains_gate_contract():
    """결과 본문에 gate: {passed, reason, holdout_size, before, after} 포함"""
    fake_s3 = FakeS3(
        objects={"feedback/metadata.json": json.dumps({"pending_count": 200})}
    )

    result = _run_pipeline(fake_s3, _metrics(0.2, 0.7), _metrics(0.1, 0.85))

    gate = result["gate"]
    assert set(["passed", "reason", "holdout_size", "before", "after"]) <= set(gate)
    assert gate["before"]["mse"] == 0.2
    assert gate["after"]["mse"] == 0.1


def test_lambda_handler_exposes_gate_in_body():
    """lambda_handler 응답 본문에서도 게이트 결과를 볼 수 있다"""

    def fake_pipeline(event):
        return {
            "success": False,
            "new_version": "v6_feedback_test",
            "model_promoted": False,
            "gate": {
                "passed": False,
                "reason": "mse_regressed",
                "holdout_size": 30,
                "before": {"mse": 0.1},
                "after": {"mse": 0.5},
            },
            "message": "Quality gate rejected: mse_regressed",
        }

    with patch.object(lf, "get_pending_count", return_value=500), patch.object(
        lf, "run_training_pipeline", side_effect=fake_pipeline
    ):
        response = lf.lambda_handler({"trigger_type": "scheduled"}, None)

    body = json.loads(response["body"])
    assert response["statusCode"] == 200
    assert body["success"] is False
    assert body["model_promoted"] is False
    assert body["gate"]["reason"] == "mse_regressed"
    assert body["gate"]["holdout_size"] == 30


def test_evaluate_holdout_returns_expected_metric_shape():
    """evaluate_holdout 이 실제 모델로 mse/ranking 을 계산한다"""
    torch.manual_seed(20260911)
    model = lf.RecommendationModelV6()

    face = np.tile(
        np.array([1.2, 450.0, 560.0, 445.0, 0.82, 0.80], dtype=np.float32), (6, 1)
    )
    skin = np.tile(np.array([79.0, 12.0], dtype=np.float32), (6, 1))
    style = np.random.RandomState(0).randn(6, 384).astype(np.float32)
    gt_norm = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    ids = ["a", "a", "b", "b", "c", "c"]

    metrics = lf.evaluate_holdout(model, face, skin, style, gt_norm, ids)

    assert metrics["num_samples"] == 6
    assert metrics["num_pairs"] == 3
    assert 0.0 <= metrics["ranking_accuracy"] <= 1.0
    assert metrics["mse"] >= 0.0
    assert metrics["mae"] >= 0.0
