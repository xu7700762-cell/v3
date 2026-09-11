from __future__ import annotations

from pathlib import Path

import torch

from ..model.main import FEMBAKANMultiTaskModel
from ..protocol import SEVERITY_THRESHOLD, STATE_DECISION
from ..training.data import FoldProtocol, load_training_dataset
from ..training.runner import (
    CHECKPOINT_SCHEMA,
    SEVERITY_WEIGHT,
    STATE_DEVIATION_MARGIN,
    STATE_DEVIATION_WEIGHT,
)
from .inference import (
    add_smoothed_state,
    attach_severity_labels,
    attach_state_labels,
    score_severity_examples,
    score_state_subjects,
)
from .io import sha256_file, write_csv, write_json
from .metrics import binary_metrics


FOLD_IDS = tuple(f"fold_{index}" for index in range(1, 6))


def _checkpoint_paths(checkpoint_root: Path) -> dict[str, Path]:
    paths = {fold_id: Path(checkpoint_root) / fold_id / "checkpoint.pt" for fold_id in FOLD_IDS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing trained checkpoints: " + ", ".join(missing))
    return paths


def _assert_empty_output(output_root: Path) -> None:
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty evaluation output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)


def _validate_checkpoint_metadata(
    payload: dict,
    dataset: str,
    fold_id: str,
    protocol: FoldProtocol,
    pretrain_sha256: str,
) -> None:
    expected = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "dataset": dataset,
        "fold_id": fold_id,
        "severity_weight": SEVERITY_WEIGHT,
        "state_deviation_weight": STATE_DEVIATION_WEIGHT,
        "state_deviation_margin": STATE_DEVIATION_MARGIN,
        "severity_task_sampling": "train_jittered_uniform_eval_locked_uniform_11",
        "severity_gradient_scope": "shared_kan_and_severity_head",
        "state_decision": STATE_DECISION,
        "severity_threshold": SEVERITY_THRESHOLD,
        "encoder_trainable": False,
        "pretrain_checkpoint_sha256": pretrain_sha256,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{fold_id} checkpoint metadata mismatch: {mismatches}")
    training_seed = payload.get("training_seed")
    if not isinstance(training_seed, int) or training_seed <= 0:
        raise RuntimeError(f"{fold_id} checkpoint has invalid training_seed")
    calibration = payload.get("severity_bias_calibration")
    if not isinstance(calibration, dict) or calibration.get("method") != (
        "source_balanced_accuracy_bias_shift_v1"
    ):
        raise RuntimeError(f"{fold_id} checkpoint severity bias calibration changed")
    base_normalization = payload.get("severity_base_normalization")
    if not isinstance(base_normalization, dict) or base_normalization.get("method") != (
        "source_only_frozen_femba_token_dynamics_zscore_v2"
    ):
        raise RuntimeError(f"{fold_id} checkpoint severity base normalization changed")
    if tuple(payload.get("source_subjects", ())) != protocol.source_subjects:
        raise RuntimeError(f"{fold_id} checkpoint source subjects changed")
    if tuple(payload.get("test_subjects", ())) != protocol.test_subjects:
        raise RuntimeError(f"{fold_id} checkpoint test subjects changed")
    if payload.get("model_spec", {}).get("name") != CHECKPOINT_SCHEMA:
        raise RuntimeError(f"{fold_id} checkpoint model spec changed")
    projection_variant = payload.get("model_spec", {}).get(
        "projection_variant", "kan"
    )
    if projection_variant not in {
        "kan",
        "mlp",
        "fractional_dog_polykan",
    }:
        raise RuntimeError(f"{fold_id} checkpoint projection variant changed")
    if payload.get("projection_variant", projection_variant) != projection_variant:
        raise RuntimeError(f"{fold_id} checkpoint projection metadata mismatch")
    cache = payload.get("pooled_cache")
    if not isinstance(cache, dict) or any(
        cache.get(key) != value
        for key, value in {
            "cache_schema": "femba_frozen_features_v2",
            "pretrain_checkpoint_sha256": pretrain_sha256,
            "encoder_mode": "eval",
            "pooling": "token_mean",
            "feature_dim": 525,
            "token_dynamics_dim": 2,
        }.items()
    ):
        raise RuntimeError(f"{fold_id} checkpoint pooled FEMBA cache metadata changed")
    if not isinstance(cache.get("data_fingerprint"), str) or not cache["data_fingerprint"]:
        raise RuntimeError(f"{fold_id} checkpoint has no pooled cache data fingerprint")
    if not isinstance(payload.get("model_state_dict"), dict):
        raise RuntimeError(f"{fold_id} checkpoint has no full model_state_dict")


def _load_model(
    checkpoint_path: Path,
    dataset: str,
    fold_id: str,
    protocol: FoldProtocol,
    pretrain_sha256: str,
    device: torch.device,
) -> tuple[FEMBAKANMultiTaskModel, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _validate_checkpoint_metadata(payload, dataset, fold_id, protocol, pretrain_sha256)
    spec = payload["model_spec"]
    model = FEMBAKANMultiTaskModel(
        latent_dim=int(spec["latent_dim"]),
        kan_degree=int(spec["kan_degree"]),
        head_dropout=float(spec["head_dropout"]),
        projection_variant=str(spec.get("projection_variant", "kan")),
        seed=int(payload["training_seed"]),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def _reference_sessions(protocol: FoldProtocol) -> dict[str, str]:
    result = {}
    for example in protocol.test_examples:
        previous = result.setdefault(example.subject_id, example.reference_session)
        if previous != example.reference_session:
            raise ValueError(f"{example.subject_id} has inconsistent reference sessions")
    if set(result) != set(protocol.test_subjects):
        raise ValueError("Every outer-test subject must have a reference session")
    return result


def run_trained_evaluation(
    config: dict,
    dataset: str,
    checkpoint_root: Path,
    output_root: Path,
    device_name: str,
) -> dict:
    if dataset not in {"monifeixing", "vrq", "city"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    checkpoints = _checkpoint_paths(Path(checkpoint_root))
    _assert_empty_output(Path(output_root))
    dataset_data = load_training_dataset(config, dataset)
    pretrain_sha256 = sha256_file(Path(config["paths"]["pretrain_checkpoint"]))
    all_state, all_severity, fold_reports = [], [], []
    training_seeds = set()
    projection_variants = set()
    for fold_id in FOLD_IDS:
        protocol = dataset_data.folds[fold_id]
        model, payload = _load_model(
            checkpoints[fold_id],
            dataset,
            fold_id,
            protocol,
            pretrain_sha256,
            device,
        )
        training_seeds.add(int(payload["training_seed"]))
        projection_variants.add(
            str(payload["model_spec"].get("projection_variant", "kan"))
        )
        state_scored = score_state_subjects(
            model,
            dataset_data.bank,
            protocol.test_subjects,
            _reference_sessions(protocol),
            device,
        )
        state = add_smoothed_state(
            attach_state_labels(state_scored, dataset_data.bank),
            3,
        )
        severity_scored = score_severity_examples(
            model,
            dataset_data.bank,
            protocol.test_examples,
            device,
            threshold=float(payload["severity_threshold"]),
        )
        severity = attach_severity_labels(severity_scored, protocol.test_examples)
        for row in state:
            row["fold_id"] = fold_id
        for row in severity:
            row["fold_id"] = fold_id
        all_state.extend(state)
        all_severity.extend(severity)
        fold_reports.append(
            {
                "fold_id": fold_id,
                "checkpoint_sha256": sha256_file(checkpoints[fold_id]),
                "source_subjects": list(protocol.source_subjects),
                "test_subjects": list(protocol.test_subjects),
                "best_epoch": int(payload["best_epoch"]),
                "training_seed": int(payload["training_seed"]),
                "projection_variant": str(
                    payload["model_spec"].get("projection_variant", "kan")
                ),
                "adaptive_basis_lr": payload.get("adaptive_basis_lr"),
                "state_decision": payload["state_decision"],
                "severity_threshold": float(payload["severity_threshold"]),
                "severity_bias_calibration": payload["severity_bias_calibration"],
                "severity_base_normalization": payload["severity_base_normalization"],
                "parameter_deltas": payload["parameter_deltas"],
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(training_seeds) != 1:
        raise RuntimeError(f"Evaluation checkpoints mix training seeds: {training_seeds}")
    if len(projection_variants) != 1:
        raise RuntimeError(
            f"Evaluation checkpoints mix projection variants: {projection_variants}"
        )
    training_seed = next(iter(training_seeds))
    projection_variant = next(iter(projection_variants))
    scored_state = [row for row in all_state if not int(row["calibration_anchor"])]
    smoothed_state = [
        {
            **row,
            "score": float(row["smoothed_score"]),
            "y_pred": int(row["smoothed_y_pred"]),
            "correct": int(row["smoothed_correct"]),
        }
        for row in scored_state
    ]
    report = {
        "status": "complete",
        "evaluation": CHECKPOINT_SCHEMA,
        "dataset": dataset,
        "protocol": {
            "raw_windows": True,
            "training_pooled_cache": "frozen_FEMBA_eval_token_mean_525_plus_token_dynamics_2",
            "encoder_trainable": False,
            "training_seed": training_seed,
            "projection_variant": projection_variant,
            "anchor_calibration": f"U3-U6 shared {projection_variant.upper()} embedding mean",
            "state_decision": STATE_DECISION,
            "severity_threshold": SEVERITY_THRESHOLD,
            "severity_bias_calibration": "source_balanced_accuracy_bias_shift_v1",
            "severity_base_normalization": "source_only_frozen_femba_token_dynamics_zscore_v2",
            "temporal_context": "[t,t+1,t+2] embedding mean within session",
            "severity_task_windows": 11,
            "severity_task_sampling": "train_jittered_uniform_eval_locked_uniform_11",
            "severity_gradient_scope": "shared_kan_and_severity_head",
            "state_smoothing": "secondary [t,t+1,t+2] within subject/session",
            "outer_labels_attached_after_scoring": True,
        },
        "state_metrics": binary_metrics(scored_state),
        "smoothed_state_metrics": binary_metrics(smoothed_state),
        "severity_metrics": binary_metrics(all_severity),
        "folds": fold_reports,
    }
    write_csv(Path(output_root) / "state_predictions.csv", all_state)
    write_csv(Path(output_root) / "severity_predictions.csv", all_severity)
    write_json(Path(output_root) / "aggregate_report.json", report)
    return report
