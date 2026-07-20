"""Regression tests for the V6 grouped-ranking training utilities."""

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


# The utilities tested here do not need the heavyweight embedding model.
_sentence_transformers = types.ModuleType("sentence_transformers")
_sentence_transformers.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", _sentence_transformers)

_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "train_model_v6_multitoken.py"
_SPEC = importlib.util.spec_from_file_location(
    "train_model_v6_multitoken", _MODULE_PATH
)
train_v6 = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(train_v6)


SAMPLES_PER_FACE = 6


def make_data(num_faces: int) -> dict:
    """Build deterministic, traceable synthetic training data."""
    face_ids = np.repeat(np.arange(num_faces, dtype=np.float32), SAMPLES_PER_FACE)
    n = len(face_ids)
    return {
        "face_features": np.column_stack(
            [face_ids, np.zeros((n, 5), dtype=np.float32)]
        ).astype(np.float32),
        "skin_features": np.zeros((n, 2), dtype=np.float32),
        "style_embeddings": np.zeros((n, 384), dtype=np.float32),
        "scores": np.tile(
            np.array([10, 25, 40, 55, 70, 95], dtype=np.float32), num_faces
        ),
    }


def test_face_group_sampler_rejects_incomplete_face_group():
    """Ranking must not silently discard samples that cannot form a face group."""
    with pytest.raises(ValueError, match="divisible|incomplete"):
        train_v6.FaceGroupBatchSampler(
            list(range(7)), samples_per_face=SAMPLES_PER_FACE
        )


def test_dataloaders_reject_split_that_would_create_empty_validation_or_test_loader():
    """Small datasets must fail clearly instead of later dividing metrics by zero."""
    with pytest.raises(ValueError, match="at least one face|val|test"):
        train_v6.create_dataloaders_no_leakage(
            make_data(3), samples_per_face=SAMPLES_PER_FACE
        )


def test_pairwise_ranking_loss_matches_manual_two_face_reverse_order_calculation():
    """A two-face reverse ranking has the hand-calculated margin loss."""
    targets = torch.tensor([[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]] * 2)
    predictions = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0] * 2, requires_grad=True)

    loss = train_v6.pairwise_ranking_loss(predictions, targets, margin=0.05)

    # For each ascending target pair i<j: signed prediction difference is
    # -(j-i)*0.2, so each loss is 0.05 + (j-i)*0.2.
    expected = sum(0.05 + (j - i) * 0.2 for i in range(6) for j in range(i + 1, 6)) / 15
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_dataloaders_keep_faces_disjoint_across_splits_and_group_validation_batches():
    """Face-level splitting prevents leakage and preserves validation ranking groups."""
    train_loader, val_loader, test_loader = train_v6.create_dataloaders_no_leakage(
        make_data(20), batch_size=64, samples_per_face=SAMPLES_PER_FACE
    )

    def ids(loader):
        return {
            int(face_id)
            for face_features, _skin, _style, _scores in loader
            for face_id in face_features[:, 0].tolist()
        }

    train_ids, val_ids, test_ids = ids(train_loader), ids(val_loader), ids(test_loader)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == set(range(20))

    for face_features, _skin, _style, _scores in val_loader:
        assert len(face_features) % SAMPLES_PER_FACE == 0
        for group in face_features[:, 0].view(-1, SAMPLES_PER_FACE):
            assert group.unique().numel() == 1


class FixedPredictionModel(torch.nn.Module):
    def __init__(self, predictions: torch.Tensor):
        super().__init__()
        self.register_buffer("predictions", predictions)
        self.offset = 0

    def forward(self, face_features, _skin_features, _style_embeddings):
        result = self.predictions[self.offset : self.offset + len(face_features)]
        self.offset += len(face_features)
        return result


def test_validate_returns_nan_ranking_metrics_but_finite_total_loss_for_short_batch():
    """The no-ranking-batch path must not poison total loss with NaN."""
    data = make_data(1)
    dataset = train_v6.HairstyleDatasetV6(
        data["face_features"][:5],
        data["skin_features"][:5],
        data["style_embeddings"][:5],
        data["scores"][:5],
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False)
    model = FixedPredictionModel(torch.zeros(5))

    metrics = train_v6.validate(
        model, loader, torch.nn.MSELoss(), torch.device("cpu"), rank_weight=1.0
    )

    assert math.isfinite(metrics["total_loss"])
    assert math.isnan(metrics["rank_loss"])
    assert math.isnan(metrics["pairwise_accuracy"])
