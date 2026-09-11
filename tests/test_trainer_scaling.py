"""
Trainer Lambda 입력 스케일링 / 라벨 정규화 / MIN_SAMPLES 게이트 회귀 테스트

대상: lambda_trainer/lambda_function.py

검증 항목:
1. trainer 의 scale_input_features 가 serving(models/ml_recommender)과 수치적으로 동일
2. FeedbackDataset 이 텐서 변환 전에 스케일링을 적용 (train/serve skew 방지)
3. evaluate_model 이 정규화된 정답(0~1)으로 호출됨
4. 실제 로드된 샘플 수 기준으로 MIN_SAMPLES 를 검사 (force=True 는 우회)

AWS 호출은 전부 목(mock) 처리한다.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import models.ml_recommender as mlr


def _import_lambda_trainer():
    """lambda_trainer/lambda_function.py 를 저장소 import 없이 로드"""
    sys.path.insert(0, "lambda_trainer")
    try:
        import lambda_function

        return lambda_function
    finally:
        sys.path.pop(0)


lf = _import_lambda_trainer()


# ========== (1) serving 과 수치적 동일성 ==========


def test_feature_stats_match_serving():
    """FACE/SKIN_FEATURE_STATS 가 serving 과 완전히 동일해야 한다"""
    assert lf.FACE_FEATURE_STATS == mlr.FACE_FEATURE_STATS
    assert lf.SKIN_FEATURE_STATS == mlr.SKIN_FEATURE_STATS


def test_scale_input_features_matches_serving():
    """랜덤 원본 벡터에 대해 trainer/serving 스케일링 결과가 동일해야 한다"""
    rng = np.random.default_rng(20260908)

    cases = []
    for _ in range(20):
        face = np.array(
            [
                rng.uniform(0.8, 1.7),  # face_ratio
                rng.uniform(60, 520),  # forehead_width (pixel)
                rng.uniform(70, 660),  # cheekbone_width (pixel)
                rng.uniform(55, 540),  # jaw_width (pixel)
                rng.uniform(0.6, 0.95),  # forehead_ratio
                rng.uniform(0.6, 0.95),  # jaw_ratio
            ],
            dtype=np.float32,
        )
        skin = np.array([rng.uniform(30, 95), rng.uniform(1, 160)], dtype=np.float32)
        cases.append((face, skin))

    # 엣지 케이스: cheekbone_width = 0 (scale_factor = 1.0 이어야 함)
    cases.append(
        (
            np.array([1.2, 400.0, 0.0, 380.0, 0.8, 0.8], dtype=np.float32),
            np.array([70.0, 15.0], dtype=np.float32),
        )
    )
    # 엣지 케이스: 이미 학습 스케일인 값 (idempotent 확인용)
    cases.append(
        (
            np.array([1.2, 458.13, 561.34, 447.7, 0.82, 0.8], dtype=np.float32),
            np.array([79.91, 12.09], dtype=np.float32),
        )
    )

    for face, skin in cases:
        trainer_face, trainer_skin = lf.scale_input_features(face.copy(), skin.copy())
        serving_face, serving_skin = mlr.scale_input_features(face.copy(), skin.copy())

        np.testing.assert_array_equal(trainer_face, serving_face)
        np.testing.assert_array_equal(trainer_skin, serving_skin)


def test_scale_input_features_is_idempotent():
    """이미 스케일링된 값에 다시 적용해도 결과가 같아야 한다"""
    face = np.array([1.2, 120.0, 150.0, 118.0, 0.8, 0.79], dtype=np.float32)
    skin = np.array([70.0, 15.0], dtype=np.float32)

    once_face, once_skin = lf.scale_input_features(face, skin)
    twice_face, twice_skin = lf.scale_input_features(once_face, once_skin)

    np.testing.assert_allclose(once_face, twice_face, rtol=1e-6)
    np.testing.assert_allclose(once_skin, twice_skin, rtol=1e-6)

    # 픽셀 특징이 학습 분포(cheekbone ~561)로 끌어올려져야 한다
    assert once_face[2] == pytest.approx(561.34, rel=1e-4)


def test_scale_feature_batch_skips_unexpected_shape():
    """레거시 one-hot 등 차원이 다른 입력은 원본을 그대로 반환"""
    face = np.zeros((3, 4), dtype=np.float32)
    skin = np.zeros((3, 2), dtype=np.float32)

    out_face, out_skin = lf.scale_feature_batch(face, skin)

    np.testing.assert_array_equal(out_face, face)
    np.testing.assert_array_equal(out_skin, skin)


# ========== (2) FeedbackDataset 스케일링 ==========


def test_feedback_dataset_scales_before_tensor_conversion():
    """FeedbackDataset 이 원본 픽셀값이 아니라 스케일링된 값을 담아야 한다"""
    face = np.array(
        [
            [1.2, 100.0, 130.0, 105.0, 0.78, 0.80],
            [1.1, 90.0, 120.0, 95.0, 0.75, 0.79],
        ],
        dtype=np.float32,
    )
    skin = np.array([[70.0, 15.0], [65.0, 20.0]], dtype=np.float32)
    style = np.zeros((2, 384), dtype=np.float32)
    gt = np.array([90.0, 10.0], dtype=np.float32)

    dataset = lf.FeedbackDataset(face, skin, style, gt)

    expected_face, expected_skin = lf.scale_feature_batch(face, skin)

    np.testing.assert_allclose(dataset.face_features.numpy(), expected_face, rtol=1e-6)
    np.testing.assert_allclose(dataset.skin_features.numpy(), expected_skin, rtol=1e-6)

    # 원본(픽셀 130) 이 그대로 들어가면 안 된다
    assert dataset.face_features[0][2].item() == pytest.approx(561.34, rel=1e-4)

    # 라벨은 기존과 동일하게 0~1 정규화
    np.testing.assert_allclose(
        dataset.ground_truths.numpy(), np.array([80.0 / 85.0, 0.0]), rtol=1e-6
    )


def test_evaluate_model_scales_inputs():
    """evaluate_model 도 동일한 스케일링을 적용한다"""
    face = np.array([[1.2, 100.0, 130.0, 105.0, 0.78, 0.80]], dtype=np.float32)
    skin = np.array([[70.0, 15.0]], dtype=np.float32)
    style = np.zeros((1, 384), dtype=np.float32)
    gt = np.array([1.0], dtype=np.float32)

    seen = {}

    class SpyModel(torch.nn.Module):
        def forward(self, face_t, skin_t, style_t):
            seen["face"] = face_t.detach().numpy().copy()
            return torch.zeros(face_t.shape[0])

    metrics = lf.evaluate_model(SpyModel(), face, skin, style, gt)

    assert seen["face"][0][2] == pytest.approx(561.34, rel=1e-4)
    assert metrics["num_samples"] == 1
    # 지표 이름이 유지되어야 한다
    for key in ("mse", "mae", "rmse", "precision", "recall", "f1_score"):
        assert key in metrics


# ========== (3)/(4) 파이프라인: 라벨 정규화 + MIN_SAMPLES 게이트 ==========


def _fake_feedbacks(count: int):
    face = np.tile(
        np.array([1.2, 100.0, 130.0, 105.0, 0.78, 0.80], dtype=np.float32), (count, 1)
    )
    skin = np.tile(np.array([70.0, 15.0], dtype=np.float32), (count, 1))
    style = np.zeros((count, 384), dtype=np.float32)
    gt = np.array([90.0 if i % 2 == 0 else 10.0 for i in range(count)], np.float32)
    keys = [f"feedback/pending/f{i}.npz" for i in range(count)]
    analysis_ids = [f"analysis-{i}" for i in range(count)]
    return face, skin, style, gt, count, keys, analysis_ids


def _patched_pipeline(count, evaluate_spy):
    """run_training_pipeline 의 외부 의존성을 전부 목으로 대체하는 컨텍스트 스택"""
    data = _fake_feedbacks(count)
    model = MagicMock()

    return [
        patch.object(lf, "load_pending_feedbacks", return_value=data),
        patch.object(lf, "load_base_model", return_value=(model, {"version": "v6"})),
        patch.object(lf, "evaluate_holdout", side_effect=evaluate_spy),
        patch.object(lf, "fine_tune_model", return_value=(model, {"final_loss": 0.12})),
        patch.object(lf, "save_model_to_s3", return_value=True),
        patch.object(lf, "save_rejected_model", return_value=True),
        patch.object(lf, "move_pending_to_processed", return_value=True),
        patch.object(lf, "update_metadata", return_value=None),
        patch.object(lf, "save_evaluation_report", return_value=True),
    ]


def test_pipeline_passes_normalized_ground_truth_to_evaluate():
    """evaluate_holdout 이 (gt - LABEL_MIN) / LABEL_RANGE 로 호출되어야 한다"""
    calls = []

    def spy(model, face, skin, style, gt, analysis_ids):
        calls.append(np.asarray(gt).copy())
        return {"mse": 0.1, "ranking_accuracy": 0.8, "num_pairs": 8}

    patches = _patched_pipeline(200, spy)
    for p in patches:
        p.start()
    try:
        result = lf.run_training_pipeline({"trigger_type": "manual"})
    finally:
        for p in patches:
            p.stop()

    assert result["success"] is True
    assert len(calls) == 2  # 학습 전/후

    raw = _fake_feedbacks(200)[3]
    analysis_ids = _fake_feedbacks(200)[6]
    _, holdout_idx = lf.split_holdout_indices(analysis_ids, lf.TRAIN_HOLDOUT_RATIO)
    expected = ((raw - lf.LABEL_MIN) / lf.LABEL_RANGE)[np.asarray(holdout_idx)]

    for observed in calls:
        # 평가는 홀드아웃 분할에 대해서만 수행된다
        np.testing.assert_allclose(observed, expected, rtol=1e-6)
        assert observed.min() >= 0.0
        assert observed.max() <= 1.0


def test_pipeline_uses_loaded_sample_count_for_min_samples():
    """메타데이터 카운터가 아니라 실제 로드된 샘플 수로 학습 시작을 막아야 한다"""
    spy_calls = []

    def spy(model, face, skin, style, gt, analysis_ids):
        spy_calls.append(gt)
        return {}

    patches = _patched_pipeline(9, spy)
    started = patches[1]  # load_base_model
    for p in patches:
        p.start()
    try:
        mock_load_base_model = lf.load_base_model
        result = lf.run_training_pipeline({"trigger_type": "scheduled"})
        assert mock_load_base_model.call_count == 0
    finally:
        for p in patches:
            p.stop()

    assert result["success"] is False
    assert "Insufficient data" in result["message"]
    assert result["loaded_count"] == 9
    assert spy_calls == []
    assert started is not None


def test_pipeline_force_bypasses_min_samples():
    """force=True 는 MIN_SAMPLES 만 우회한다 (학습은 시작되지만 게이트는 살아있다)"""

    def spy(model, face, skin, style, gt, analysis_ids):
        return {"mse": 0.1, "ranking_accuracy": 0.8, "num_pairs": 4}

    patches = _patched_pipeline(9, spy)
    for p in patches:
        p.start()
    try:
        result = lf.run_training_pipeline({"trigger_type": "manual", "force": True})
        # MIN_SAMPLES 를 넘겨 실제로 학습까지 진입한다
        assert lf.load_base_model.call_count == 1
        assert lf.fine_tune_model.call_count == 1
    finally:
        for p in patches:
            p.stop()

    assert result["samples_trained"] == 9
    # 9건으로는 홀드아웃이 부족하므로 force 여도 모델은 교체되지 않는다
    assert result["gate"]["passed"] is False
    assert result["gate"]["reason"] == "insufficient_holdout"
    assert result["model_promoted"] is False


def test_pipeline_force_with_skip_gate_promotes():
    """게이트 우회는 force 가 아니라 skip_gate 플래그로만 가능하다"""

    def spy(model, face, skin, style, gt, analysis_ids):
        return {"mse": 0.1, "ranking_accuracy": 0.8, "num_pairs": 4}

    patches = _patched_pipeline(9, spy)
    for p in patches:
        p.start()
    try:
        result = lf.run_training_pipeline(
            {"trigger_type": "manual", "force": True, "skip_gate": True}
        )
    finally:
        for p in patches:
            p.stop()

    assert result["success"] is True
    assert result["gate"]["passed"] is True
    assert result["gate"]["reason"] == "skipped"
    assert result["model_promoted"] is True


def test_trainer_has_no_repo_imports():
    """trainer 는 self-contained 여야 한다 (저장소 모듈 import 금지)"""
    import io as _io

    with _io.open("lambda_trainer/lambda_function.py", encoding="utf-8") as f:
        source = f.read()

    for forbidden in (
        "from models.",
        "import models.",
        "from services.",
        "from utils.",
    ):
        assert forbidden not in source, forbidden
