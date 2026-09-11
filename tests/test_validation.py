from pathlib import Path

import numpy as np
import pytest

from vestibular_fusion.data import vrq
from vestibular_fusion.data.features import fit_and_apply_subject_ea
from vestibular_fusion.evaluation.io import sha256_file
from vestibular_fusion.evaluation.metrics import binary_metrics
from vestibular_fusion.evaluation.trained import _validate_checkpoint_metadata
from vestibular_fusion.preflight import _check_fivefold_subject_split, _require_hash
from vestibular_fusion.training.data import FoldProtocol
from vestibular_fusion.training.runner import (
    CHECKPOINT_SCHEMA,
    SEVERITY_WEIGHT,
    STATE_DEVIATION_MARGIN,
    STATE_DEVIATION_WEIGHT,
)


def _valid_folds():
    return {
        f"fold_{index}": {
            "train_subjects": [f"train_{index}"],
            "val_subjects": [f"val_{index}"],
            "test_subjects": [f"test_{index}"],
        }
        for index in range(1, 6)
    }


def test_fivefold_subject_split_is_complete_and_disjoint():
    _check_fivefold_subject_split({"folds": _valid_folds()}, "synthetic manifest")


def test_fivefold_subject_overlap_fails_loudly():
    folds = _valid_folds()
    folds["fold_2"]["test_subjects"] = ["test_1"]
    with pytest.raises(RuntimeError, match="test subjects overlap"):
        _check_fivefold_subject_split({"folds": folds}, "synthetic manifest")


def test_source_train_validation_overlap_fails_loudly():
    folds = _valid_folds()
    folds["fold_1"]["val_subjects"] = ["train_1"]
    with pytest.raises(RuntimeError, match="source-train/source-val subjects overlap"):
        _check_fivefold_subject_split({"folds": folds}, "synthetic manifest")


def test_binary_metrics_use_predictions_and_scores():
    metrics = binary_metrics(
        [
            {"y_true": 0, "y_pred": 0, "score": 0.1, "subject_id": "s1"},
            {"y_true": 1, "y_pred": 1, "score": 0.9, "subject_id": "s2"},
        ]
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["confusion_matrix"] == [[1, 0], [0, 1]]


def test_checkpoint_requires_the_saved_source_validation_threshold():
    protocol = FoldProtocol(("s1", "s2"), ("s3",), ("s1",), ("s2",), (), ())
    payload = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "dataset": "vrq",
        "fold_id": "fold_1",
        "training_seed": 1001,
        "severity_weight": SEVERITY_WEIGHT,
        "state_deviation_weight": STATE_DEVIATION_WEIGHT,
        "state_deviation_margin": STATE_DEVIATION_MARGIN,
        "severity_task_sampling": "train_jittered_uniform_eval_locked_uniform_11",
        "severity_gradient_scope": "shared_kan_and_severity_head",
        "encoder_trainable": False,
        "pretrain_checkpoint_sha256": "abc",
        "source_subjects": ["s1", "s2"],
        "test_subjects": ["s3"],
        "state_decision": "anchor_oriented_subject_kmeans_160d",
        "severity_threshold": 0.5,
        "severity_bias_calibration": {
            "method": "source_balanced_accuracy_bias_shift_v1"
        },
        "severity_base_normalization": {
            "method": "source_only_frozen_femba_token_dynamics_zscore_v2"
        },
        "pooled_cache": {
            "cache_schema": "femba_frozen_features_v2",
            "pretrain_checkpoint_sha256": "abc",
            "data_fingerprint": "data-sha",
            "encoder_mode": "eval",
            "pooling": "token_mean",
            "feature_dim": 525,
            "token_dynamics_dim": 2,
        },
        "model_spec": {"name": CHECKPOINT_SCHEMA},
        "model_state_dict": {},
    }
    _validate_checkpoint_metadata(payload, "vrq", "fold_1", protocol, "abc")
    changed = {**payload, "state_decision": "global_threshold"}
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        _validate_checkpoint_metadata(changed, "vrq", "fold_1", protocol, "abc")


def test_sha256_validation_fails_on_changed_asset(tmp_path: Path):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"locked")
    expected = sha256_file(path)
    _require_hash(path, expected, "synthetic asset")
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _require_hash(path, expected, "synthetic asset")


def test_vrq_short_record_fails_with_domain_error(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        vrq,
        "loadmat",
        lambda path, variable_names: {
            "data256": np.zeros((32, vrq.WINDOW_SIZE - 1), dtype=np.float32)
        },
    )
    with pytest.raises(ValueError, match="no complete five-second windows"):
        vrq.load_windows(tmp_path / "short.mat", "data256")


def test_subject_ea_stays_real_and_finite_for_rank_deficient_covariance():
    base = np.random.RandomState(7).randn(3, 1, 1280).astype(np.float32)
    windows = np.repeat(base, 30, axis=1)
    aligned, transform, diagnostics = fit_and_apply_subject_ea(windows)
    assert np.isrealobj(transform)
    assert np.isfinite(transform).all()
    assert np.isfinite(aligned).all()
    assert diagnostics["ridge"] > 0.0
