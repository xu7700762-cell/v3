from types import SimpleNamespace

import numpy as np
import torch

from vestibular_fusion.data.types import AuditMetadata, FeatureBank, StateSample, SubjectRecord
from vestibular_fusion.evaluation.inference import (
    add_smoothed_state,
    apply_anchor_oriented_embedding_kmeans,
    attach_severity_labels,
    attach_state_labels,
    apply_state_threshold,
    score_severity_examples,
    score_state_subjects,
)
from vestibular_fusion.model.main import FEMBAKANMultiTaskModel
from vestibular_fusion.evaluation.metrics import select_balanced_accuracy_threshold
from vestibular_fusion.training.data import SeverityExample

from test_model import FakeEncoder


def _bank() -> FeatureBank:
    windows = np.random.RandomState(1).randn(27, 30, 1280).astype(np.float16)
    record = SubjectRecord(
        windows=windows,
        labels=np.asarray([0] * 16 + [1] * 11, dtype=np.int64),
        sessions=["rest"] * 16 + ["task"] * 11,
    )
    samples = [
        StateSample(index, "s1", record.sessions[index], int(record.labels[index]), index if index < 16 else index - 16, index, "x.mat")
        for index in range(27)
    ]
    return FeatureBank({"s1": record}, samples, AuditMetadata({}))


def test_scores_are_generated_before_labels_are_attached():
    model = FEMBAKANMultiTaskModel(encoder=FakeEncoder(), seed=3)
    bank = _bank()
    state_scored = score_state_subjects(
        model, bank, ("s1",), {"s1": "rest"}, torch.device("cpu")
    )
    assert "y_true" not in state_scored[0]
    state = attach_state_labels(state_scored, bank)
    assert "y_true" in state[0]
    assert sum(row["calibration_anchor"] for row in state) == 4
    assert "smoothed_score" in add_smoothed_state(state)[0]

    examples = (SeverityExample("s1", "rest", "task", 1),)
    severity_scored = score_severity_examples(
        model, bank, examples, torch.device("cpu")
    )
    assert "y_true" not in severity_scored[0]
    severity = attach_severity_labels(severity_scored, examples)
    assert severity[0]["y_true"] == 1


def test_subject_record_has_no_token_cache():
    assert not hasattr(_bank().records["s1"], "tokens")


def test_anchor_oriented_embedding_kmeans_uses_no_labels_and_orients_the_rest_cluster():
    rows = [
        {
            "sample_index": index,
            "subject_id": "s1",
            "local_index": index,
            "calibration_anchor": int(index < 4),
            "logit": logit,
            "score": float(torch.sigmoid(torch.tensor(logit))),
        }
        for index, logit in enumerate((-1.2, -1.1, -1.0, -0.9, -0.8, -0.7, 0.9, 1.0))
    ]
    values = np.zeros((8, 160), dtype=np.float32)
    values[:6, 0] = np.asarray([-1.2, -1.1, -1.0, -0.9, -0.8, -0.7])
    values[6:, 0] = np.asarray([0.9, 1.0])
    calibrated = apply_anchor_oriented_embedding_kmeans(rows, {"s1": values})
    assert all("y_true" not in row for row in calibrated)
    assert [row["y_pred"] for row in calibrated[:6]] == [0, 0, 0, 0, 0, 0]
    assert [row["y_pred"] for row in calibrated[6:]] == [1, 1]


def test_state_threshold_is_selected_only_from_labeled_validation_rows():
    rows = [
        {"score": 0.10, "y_true": 0},
        {"score": 0.40, "y_true": 0},
        {"score": 0.35, "y_true": 1},
        {"score": 0.80, "y_true": 1},
    ]
    selection = select_balanced_accuracy_threshold(rows)
    assert selection["method"] == "max_balanced_accuracy"
    assert selection["threshold"] == 0.35
    assert selection["balanced_accuracy"] == 0.75
    calibrated = apply_state_threshold(rows, selection["threshold"])
    assert [row["y_pred"] for row in calibrated] == [0, 1, 1, 1]


def test_smoothed_metric_excludes_calibration_anchors():
    rows = [
        {
            "sample_index": index,
            "subject_id": "s1",
            "session": "rest",
            "window_index": index,
            "score": score,
            "y_true": 0,
            "calibration_anchor": anchor,
        }
        for index, score, anchor in ((0, 0.0, 0), (1, 1.0, 1), (2, 0.0, 0), (3, 0.0, 0))
    ]
    smoothed = add_smoothed_state(rows)
    assert smoothed[0]["smoothed_score"] == 0.0
    assert smoothed[1]["smoothed_score"] is None
