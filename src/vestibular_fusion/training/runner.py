from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "2001")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

from ..data.anchors import (
    anchor_indices,
    context_features,
    jittered_task_indices,
    task_indices,
)
from ..evaluation.inference import (
    attach_severity_labels,
    attach_state_labels,
    score_severity_examples,
    score_state_subjects,
)
from ..evaluation.io import sha256_file, write_json
from ..evaluation.metrics import binary_metrics
from ..model.main import FEMBAKANMultiTaskModel
from ..protocol import SEVERITY_THRESHOLD, STATE_DECISION
from .cache import PooledFeatureCache, load_or_build_pooled_cache
from .data import FoldProtocol, SeverityExample, load_training_dataset


CHECKPOINT_SCHEMA = "femba_kan_mtl_v27"
SEVERITY_WEIGHT = 0.5
STATE_DEVIATION_WEIGHT = 0.3
STATE_DEVIATION_MARGIN = 0.25
DEFAULT_TRAINING_SEED = 2001
MAX_EPOCHS = 60
PATIENCE = 15
MIN_EPOCHS = 20
DOMAINS_PER_BATCH = 5
TRIALS_PER_CLASS = 5
SEVERITY_BATCH_SIZE = 4
HEAD_LR = 2e-5
ADAPTIVE_BASIS_LR = 2e-4
SEVERITY_HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0
AMP_DTYPE = torch.bfloat16
LABEL_SMOOTHING = 0.05


def _seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _training_seed(config: dict) -> int:
    seed = int(config.get("training_seed", DEFAULT_TRAINING_SEED))
    if seed <= 0:
        raise ValueError("training_seed must be positive")
    return seed


def _reference_sessions(examples: tuple[SeverityExample, ...]) -> dict[str, str]:
    result = {}
    for example in examples:
        previous = result.setdefault(example.subject_id, example.reference_session)
        if previous != example.reference_session:
            raise ValueError(f"{example.subject_id} has inconsistent reference sessions")
    return result


def _state_batches(
    bank,
    subjects: tuple[str, ...],
    reference_sessions: dict[str, str],
    seed: int,
):
    by_subject = {}
    for subject in subjects:
        record = bank.records[subject]
        anchors = set(int(index) for index in anchor_indices(record, reference_sessions[subject]))
        by_label = {
            label: [
                index
                for index, value in enumerate(record.labels)
                if int(value) == label and index not in anchors
            ]
            for label in (0, 1)
        }
        if any(len(values) < TRIALS_PER_CLASS for values in by_label.values()):
            raise ValueError(f"{subject} cannot provide balanced non-anchor state windows")
        by_subject[subject] = by_label
    rng = random.Random(int(seed))
    total = sum(sum(len(values) for values in item.values()) for item in by_subject.values())
    batches = math.ceil(total / (DOMAINS_PER_BATCH * TRIALS_PER_CLASS * 2))
    ordered_subjects = sorted(by_subject)
    for _ in range(batches):
        chosen = rng.sample(ordered_subjects, min(DOMAINS_PER_BATCH, len(ordered_subjects)))
        while len(chosen) < DOMAINS_PER_BATCH:
            chosen.append(rng.choice(ordered_subjects))
        local_indices = []
        for subject in chosen:
            indices = rng.sample(by_subject[subject][0], TRIALS_PER_CLASS) + rng.sample(
                by_subject[subject][1], TRIALS_PER_CLASS
            )
            rng.shuffle(indices)
            local_indices.append(indices)
        yield chosen, local_indices


def _state_tensors(
    bank,
    pooled_features: dict[str, np.ndarray],
    subjects: list[str],
    local_indices: list[list[int]],
    references: dict,
):
    windows, anchors, labels = [], [], []
    for subject, indices in zip(subjects, local_indices):
        record = bank.records[subject]
        features = pooled_features[subject]
        windows.append(context_features(record, features, indices))
        anchors.append(
            context_features(
                record, features, anchor_indices(record, references[subject])
            )
        )
        labels.append(record.labels[indices])
    return (
        torch.from_numpy(np.stack(windows).astype(np.float32)),
        torch.from_numpy(np.stack(anchors).astype(np.float32)),
        torch.from_numpy(np.stack(labels).astype(np.float32)),
    )


def _severity_tensors(
    bank,
    pooled_features: dict[str, np.ndarray],
    token_dynamics: dict[str, np.ndarray],
    examples: list[SeverityExample],
    *,
    jitter_seed: int | None = None,
):
    anchors, tasks, task_dynamics, labels = [], [], [], []
    for offset, example in enumerate(examples):
        record = bank.records[example.subject_id]
        features = pooled_features[example.subject_id]
        dynamics = token_dynamics[example.subject_id]
        anchors.append(
            context_features(
                record, features, anchor_indices(record, example.reference_session)
            )
        )
        indices = (
            task_indices(record, example.task_session, 11)
            if jitter_seed is None
            else jittered_task_indices(
                record,
                example.task_session,
                11,
                int(jitter_seed) + offset * 1009,
            )
        )
        tasks.append(context_features(record, features, indices))
        task_dynamics.append(context_features(record, dynamics, indices))
        labels.append(float(example.label))
    return (
        torch.from_numpy(np.stack(tasks).astype(np.float32)),
        torch.from_numpy(np.stack(task_dynamics).astype(np.float32)),
        torch.from_numpy(np.stack(anchors).astype(np.float32)),
        torch.tensor(labels, dtype=torch.float32),
    )


def _severity_base_normalization(
    model: FEMBAKANMultiTaskModel,
    bank,
    pooled_features: dict[str, np.ndarray],
    token_dynamics: dict[str, np.ndarray],
    examples: tuple[SeverityExample, ...],
) -> dict:
    values = []
    for example in examples:
        record = bank.records[example.subject_id]
        contexts = context_features(
            record,
            pooled_features[example.subject_id],
            task_indices(record, example.task_session, 11),
        )
        task = contexts.mean(axis=1, dtype=np.float64)
        dynamics_contexts = context_features(
            record,
            token_dynamics[example.subject_id],
            task_indices(record, example.task_session, 11),
        )
        task_dynamics = dynamics_contexts.mean(axis=1, dtype=np.float64)
        with torch.no_grad():
            base = model.severity_head.base_feature(
                torch.from_numpy(task.astype(np.float32))[None],
                torch.from_numpy(task_dynamics.astype(np.float32))[None],
            )[0]
        values.append(base.double().numpy())
    array = np.stack(values).astype(np.float64, copy=False)
    center = array.mean(axis=0)
    scale = array.std(axis=0)
    if not np.isfinite(array).all() or np.any(scale < 1e-8):
        raise ValueError("Source FEMBA token dynamics have no usable variation")
    model.severity_head.set_base_normalization(center.tolist(), scale.tolist())
    return {
        "method": "source_only_frozen_femba_token_dynamics_zscore_v2",
        "source_examples": int(len(array)),
        "center": center.tolist(),
        "scale": scale.tolist(),
    }


def _optimizer(model: FEMBAKANMultiTaskModel) -> torch.optim.AdamW:
    adaptive_basis_names = {
        "shared_kan.log_input_scale",
        "shared_kan.fractional_order_logit",
        "shared_kan.rational_logits",
        "shared_kan.order_logits",
        "shared_kan.linear_residual_logit",
        "shared_kan.dog_scale_logit",
        "shared_kan.dog_shift",
        "shared_kan.dog_mix_logits",
    }
    adaptive_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if model.projection_variant == "fractional_dog_polykan"
        and name in adaptive_basis_names
    ]
    adaptive_ids = {id(parameter) for parameter in adaptive_parameters}
    shared_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("severity_head.")
        and id(parameter) not in adaptive_ids
    ]
    severity_parameters = list(model.severity_head.parameters())
    if not shared_parameters or not severity_parameters:
        raise AssertionError("Optimizer requires shared KAN/state and severity parameters")
    groups = [{"params": shared_parameters, "lr": HEAD_LR}]
    if adaptive_parameters:
        groups.append(
            {
                "params": adaptive_parameters,
                "lr": float(getattr(model, "adaptive_basis_lr", ADAPTIVE_BASIS_LR)),
            }
        )
    groups.append({"params": severity_parameters, "lr": SEVERITY_HEAD_LR})
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.named_parameters()
    }


def _delta(module: torch.nn.Module, before: dict[str, torch.Tensor]) -> float:
    values = [
        (value.detach().cpu() - before[name]).reshape(-1).float()
        for name, value in module.named_parameters()
        if value.is_floating_point()
    ]
    return float(torch.linalg.vector_norm(torch.cat(values)))


def _grad_norm(module: torch.nn.Module) -> float:
    values = [
        parameter.grad.detach().float().reshape(-1)
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return float(torch.linalg.vector_norm(torch.cat(values))) if values else 0.0


def _state_deviation_loss(
    deviation: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    if deviation.shape != labels.shape:
        raise ValueError("State deviation values and labels must have the same shape")
    if deviation.ndim == 1:
        deviation = deviation.unsqueeze(0)
        labels = labels.unsqueeze(0)
    if deviation.ndim != 2:
        raise ValueError("State deviation learning expects [subjects,windows]")
    losses = []
    for subject_deviation, subject_labels in zip(deviation, labels):
        normal = subject_deviation[subject_labels < 0.5]
        state = subject_deviation[subject_labels > 0.5]
        if normal.numel() == 0 or state.numel() == 0:
            raise ValueError("Every subject requires both state classes")
        losses.append(
            F.softplus(
                normal[:, None] - state[None, :] + STATE_DEVIATION_MARGIN
            ).mean()
        )
    return torch.stack(losses).mean()


def _severity_bias_calibration(
    model: FEMBAKANMultiTaskModel,
    bank,
    pooled_features: dict[str, np.ndarray],
    token_dynamics: dict[str, np.ndarray],
    examples: tuple[SeverityExample, ...],
    device: torch.device,
) -> dict:
    rows = attach_severity_labels(
        score_severity_examples(
            model,
            bank,
            examples,
            device,
            pooled_features=pooled_features,
            token_dynamics=token_dynamics,
        ),
        examples,
    )
    logits = np.asarray([float(row["logit"]) for row in rows], dtype=np.float64)
    labels = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Severity bias calibration requires both source classes")
    unique = np.unique(logits)
    candidates = np.concatenate(
        (
            [np.nextafter(unique[0], -np.inf)],
            (unique[:-1] + unique[1:]) / 2.0,
            [np.nextafter(unique[-1], np.inf)],
        )
    )
    best = None
    for threshold in candidates:
        predictions = logits >= float(threshold)
        sensitivity = float(predictions[labels == 1].mean())
        specificity = float((~predictions[labels == 0]).mean())
        balanced_accuracy = 0.5 * (sensitivity + specificity)
        accuracy = float((predictions == labels).mean())
        key = (balanced_accuracy, accuracy, -abs(float(threshold)))
        if best is None or key > best[0]:
            best = (key, float(threshold), sensitivity, specificity)
    if best is None:
        raise AssertionError("Severity bias calibration produced no threshold")
    _, threshold, sensitivity, specificity = best
    with torch.no_grad():
        model.severity_head.output_bias.sub_(threshold)
    return {
        "method": "source_balanced_accuracy_bias_shift_v1",
        "source_examples": int(len(rows)),
        "logit_shift": float(threshold),
        "source_balanced_accuracy": float(0.5 * (sensitivity + specificity)),
        "source_sensitivity": sensitivity,
        "source_specificity": specificity,
    }


def _new_model(config: dict, device: torch.device) -> tuple[FEMBAKANMultiTaskModel, dict]:
    projection_variant = str(
        config.get("projection_variant", "fractional_dog_polykan")
    )
    model, load_info = FEMBAKANMultiTaskModel.from_pretrained(
        config["paths"]["pretrain_checkpoint"],
        kan_degree=int(config.get("kan_degree", 2)),
        projection_variant=projection_variant,
        seed=_training_seed(config),
    )
    model.adaptive_basis_lr = float(
        config.get("adaptive_basis_lr", ADAPTIVE_BASIS_LR)
    )
    if model.adaptive_basis_lr <= 0.0:
        raise ValueError("adaptive_basis_lr must be positive")
    return model.to(device), load_info


def _train_epoch(
    model: FEMBAKANMultiTaskModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    bank,
    pooled_features: dict[str, np.ndarray],
    token_dynamics: dict[str, np.ndarray],
    training_seed: int,
    subjects: tuple[str, ...],
    examples: tuple[SeverityExample, ...],
    epoch: int,
    device: torch.device,
    *,
    max_steps: int | None = None,
) -> dict:
    started = time.perf_counter()
    if not examples:
        raise ValueError("Multitask training requires severity examples")
    references = _reference_sessions(examples)
    rng = random.Random(int(training_seed) + int(epoch) * 200003)
    severity_by_label = {
        label: [example for example in examples if int(example.label) == label]
        for label in (0, 1)
    }
    if any(not values for values in severity_by_label.values()):
        raise ValueError("Multitask training requires both severity classes")
    for values in severity_by_label.values():
        rng.shuffle(values)
    severity_per_class = SEVERITY_BATCH_SIZE // 2
    model.train()
    totals = {
        "loss": 0.0,
        "state_loss": 0.0,
        "state_deviation_loss": 0.0,
        "severity_loss": 0.0,
    }
    maxima = {"encoder": 0.0, "kan": 0.0, "state_head": 0.0, "severity_head": 0.0}
    steps = 0
    batches = _state_batches(
        bank, subjects, references, int(training_seed) + int(epoch) * 100003
    )
    for state_subjects, local_indices in batches:
        current_examples = []
        for label in (0, 1):
            values = severity_by_label[label]
            start = (steps * severity_per_class) % len(values)
            current_examples.extend(
                values[(start + offset) % len(values)]
                for offset in range(severity_per_class)
            )
        rng.shuffle(current_examples)
        state_windows, state_anchors, state_labels = _state_tensors(
            bank, pooled_features, state_subjects, local_indices, references
        )
        (
            task_windows,
            task_token_dynamics,
            severity_anchors,
            severity_labels,
        ) = _severity_tensors(
            bank,
            pooled_features,
            token_dynamics,
            current_examples,
            jitter_seed=int(training_seed) + int(epoch) * 300007 + steps * 100003,
        )
        state_windows = state_windows.to(device, non_blocking=True)
        state_anchors = state_anchors.to(device, non_blocking=True)
        state_labels = state_labels.to(device, non_blocking=True)
        task_windows = task_windows.to(device, non_blocking=True)
        task_token_dynamics = task_token_dynamics.to(device, non_blocking=True)
        severity_anchors = severity_anchors.to(device, non_blocking=True)
        severity_labels = severity_labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=AMP_DTYPE, enabled=device.type == "cuda"
        ):
            state_logits, state_deviation = model.state_logits_and_deviation_from_pooled(
                state_windows, state_anchors
            )
            severity_logits = model.severity_logits_from_pooled(
                task_windows, severity_anchors, task_token_dynamics
            )
            state_targets = state_labels * (1.0 - 2.0 * LABEL_SMOOTHING) + LABEL_SMOOTHING
            severity_targets = (
                severity_labels * (1.0 - 2.0 * LABEL_SMOOTHING) + LABEL_SMOOTHING
            )
            state_loss = F.binary_cross_entropy_with_logits(state_logits, state_targets)
            state_deviation_loss = _state_deviation_loss(
                state_deviation, state_labels
            )
            severity_loss = F.binary_cross_entropy_with_logits(
                severity_logits, severity_targets
            )
            loss = (
                state_loss
                + STATE_DEVIATION_WEIGHT * state_deviation_loss
                + SEVERITY_WEIGHT * severity_loss
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite multitask loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = {
            "encoder": _grad_norm(model.encoder),
            "kan": _grad_norm(model.shared_kan),
            "state_head": _grad_norm(model.state_head),
            "severity_head": _grad_norm(model.severity_head),
        }
        if gradients["encoder"] != 0.0 or min(
            gradients[name] for name in ("kan", "state_head", "severity_head")
        ) <= 0.0:
            raise AssertionError(
                f"Encoder must stay frozen and all heads must receive gradients: {gradients}"
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        totals["loss"] += float(loss.detach().cpu())
        totals["state_loss"] += float(state_loss.detach().cpu())
        totals["state_deviation_loss"] += float(state_deviation_loss.detach().cpu())
        totals["severity_loss"] += float(severity_loss.detach().cpu())
        for key, value in gradients.items():
            maxima[key] = max(maxima[key], value)
        steps += 1
        if max_steps is not None and steps >= int(max_steps):
            break
    if steps == 0:
        raise AssertionError("Training epoch produced no optimization steps")
    return {
        **{key: value / steps for key, value in totals.items()},
        "steps": steps,
        "elapsed_seconds": float(time.perf_counter() - started),
        **{f"max_{key}_grad_norm": value for key, value in maxima.items()},
    }


def _validation(
    model: FEMBAKANMultiTaskModel,
    bank,
    pooled_features: dict[str, np.ndarray],
    token_dynamics: dict[str, np.ndarray],
    subjects: tuple[str, ...],
    examples: tuple[SeverityExample, ...],
    device: torch.device,
) -> dict:
    started = time.perf_counter()
    references = _reference_sessions(examples)
    state_scored = score_state_subjects(
        model,
        bank,
        subjects,
        references,
        device,
        pooled_features=pooled_features,
    )
    state_rows = [
        row
        for row in attach_state_labels(state_scored, bank)
        if not int(row["calibration_anchor"])
    ]
    severity_scored = score_severity_examples(
        model,
        bank,
        examples,
        device,
        pooled_features=pooled_features,
        token_dynamics=token_dynamics,
    )
    severity_rows = attach_severity_labels(severity_scored, examples)
    state_logits = torch.tensor([row["logit"] for row in state_rows], dtype=torch.float32)
    state_labels = torch.tensor([row["y_true"] for row in state_rows], dtype=torch.float32)
    severity_logits = torch.tensor(
        [row["logit"] for row in severity_rows], dtype=torch.float32
    )
    severity_labels = torch.tensor(
        [row["y_true"] for row in severity_rows], dtype=torch.float32
    )
    state_loss = float(F.binary_cross_entropy_with_logits(state_logits, state_labels))
    severity_loss = float(
        F.binary_cross_entropy_with_logits(severity_logits, severity_labels)
    )
    return {
        "loss": state_loss + SEVERITY_WEIGHT * severity_loss,
        "elapsed_seconds": float(time.perf_counter() - started),
        "state_loss": state_loss,
        "severity_loss": severity_loss,
        "state_metrics": binary_metrics(state_rows),
        "severity_metrics": binary_metrics(severity_rows),
    }


def _select_epoch(
    config: dict,
    dataset: str,
    fold_id: str,
    bank,
    pooled_cache: PooledFeatureCache,
    protocol: FoldProtocol,
    device: torch.device,
    output_root: Path,
) -> tuple[int, dict]:
    training_seed = _training_seed(config)
    _seed_everything(training_seed)
    model, load_info = _new_model(config, device)
    training_examples = tuple(
        example
        for example in protocol.source_examples
        if example.subject_id in set(protocol.source_train_subjects)
    )
    severity_base_normalization = _severity_base_normalization(
        model,
        bank,
        pooled_cache.features,
        pooled_cache.token_dynamics,
        training_examples,
    )
    optimizer = _optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_epoch, best_loss, stale = 0, math.inf, 0
    best_key = (-math.inf, -math.inf, -math.inf)
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        training = _train_epoch(
            model,
            optimizer,
            scaler,
            bank,
            pooled_cache.features,
            pooled_cache.token_dynamics,
            training_seed,
            protocol.source_train_subjects,
            training_examples,
            epoch,
            device,
        )
        validation_examples = tuple(
            example
            for example in protocol.source_examples
            if example.subject_id in set(protocol.source_val_subjects)
        )
        validation = _validation(
            model,
            bank,
            pooled_cache.features,
            pooled_cache.token_dynamics,
            protocol.source_val_subjects,
            validation_examples,
            device,
        )
        joint_balanced_accuracy = 0.5 * (
            float(validation["state_metrics"]["balanced_accuracy"])
            + float(validation["severity_metrics"]["balanced_accuracy"])
        )
        joint_auroc = 0.5 * (
            float(validation["state_metrics"]["AUROC"])
            + float(validation["severity_metrics"]["AUROC"])
        )
        selection_key = (
            joint_balanced_accuracy,
            joint_auroc,
            -float(validation["loss"]),
        )
        eligible = epoch >= MIN_EPOCHS
        if eligible and selection_key > best_key:
            best_epoch = epoch
            best_loss = float(validation["loss"])
            best_key = selection_key
            stale = 0
        elif eligible:
            stale += 1
        row = {
            "epoch": epoch,
            "training": training,
            "validation": validation,
            "selection_key": list(selection_key),
            "best_epoch": best_epoch,
            "patience": stale,
        }
        history.append(row)
        write_json(output_root / "selection_history.json", history)
        print(
            f"{dataset}/{fold_id} epoch={epoch} train={training['loss']:.5f} "
            f"val={validation['loss']:.5f} best={best_epoch}",
            flush=True,
        )
        if eligible and stale >= PATIENCE:
            break
    if best_epoch <= 0:
        raise RuntimeError(f"{fold_id} failed to select an epoch")
    selected = next(row for row in history if int(row["epoch"]) == best_epoch)
    report = {
        "best_epoch": best_epoch,
        "training_seed": training_seed,
        "projection_variant": model.projection_variant,
        "adaptive_basis_lr": model.adaptive_basis_lr,
        "best_validation_loss": best_loss,
        "best_validation_key": list(best_key),
        "min_epochs": MIN_EPOCHS,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "pretrain_load_info": load_info,
        "pooled_cache": pooled_cache.audit(),
        "severity_base_normalization": severity_base_normalization,
        "history": history,
    }
    write_json(output_root / "selection_report.json", report)
    return best_epoch, report


def _refit(
    config: dict,
    dataset: str,
    fold_id: str,
    bank,
    pooled_cache: PooledFeatureCache,
    protocol: FoldProtocol,
    best_epoch: int,
    device: torch.device,
    output_root: Path,
) -> dict:
    training_seed = _training_seed(config)
    _seed_everything(training_seed)
    model, load_info = _new_model(config, device)
    severity_base_normalization = _severity_base_normalization(
        model,
        bank,
        pooled_cache.features,
        pooled_cache.token_dynamics,
        protocol.source_examples,
    )
    before = {
        "encoder": _snapshot(model.encoder),
        "kan": _snapshot(model.shared_kan),
        "state_head": _snapshot(model.state_head),
        "severity_head": _snapshot(model.severity_head),
    }
    optimizer = _optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(1, int(best_epoch) + 1):
        history.append(
            {
                "epoch": epoch,
                **_train_epoch(
                    model,
                    optimizer,
                    scaler,
                    bank,
                    pooled_cache.features,
                    pooled_cache.token_dynamics,
                    training_seed,
                    protocol.source_subjects,
                    protocol.source_examples,
                    epoch,
                    device,
                ),
            }
        )
        write_json(output_root / "refit_history.json", history)
    severity_bias_calibration = _severity_bias_calibration(
        model,
        bank,
        pooled_cache.features,
        pooled_cache.token_dynamics,
        protocol.source_examples,
        device,
    )
    deltas = {
        "encoder_delta_l2": _delta(model.encoder, before["encoder"]),
        "kan_delta_l2": _delta(model.shared_kan, before["kan"]),
        "state_head_delta_l2": _delta(model.state_head, before["state_head"]),
        "severity_head_delta_l2": _delta(model.severity_head, before["severity_head"]),
    }
    if deltas["encoder_delta_l2"] != 0.0 or min(
        deltas[name]
        for name in (
            "kan_delta_l2",
            "state_head_delta_l2",
            "severity_head_delta_l2",
        )
    ) <= 0.0:
        raise AssertionError(f"Refit freeze/update audit failed: {deltas}")
    checkpoint = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "dataset": dataset,
        "fold_id": fold_id,
        "training_seed": training_seed,
        "projection_variant": model.projection_variant,
        "adaptive_basis_lr": model.adaptive_basis_lr,
        "severity_weight": SEVERITY_WEIGHT,
        "state_deviation_weight": STATE_DEVIATION_WEIGHT,
        "state_deviation_margin": STATE_DEVIATION_MARGIN,
        "severity_task_sampling": "train_jittered_uniform_eval_locked_uniform_11",
        "severity_gradient_scope": "shared_kan_and_severity_head",
        "state_decision": STATE_DECISION,
        "severity_threshold": SEVERITY_THRESHOLD,
        "severity_bias_calibration": severity_bias_calibration,
        "severity_base_normalization": severity_base_normalization,
        "best_epoch": int(best_epoch),
        "source_subjects": list(protocol.source_subjects),
        "test_subjects": list(protocol.test_subjects),
        "source_train_subjects": list(protocol.source_train_subjects),
        "source_val_subjects": list(protocol.source_val_subjects),
        "pretrain_checkpoint_sha256": sha256_file(
            Path(config["paths"]["pretrain_checkpoint"])
        ),
        "pretrain_load_info": load_info,
        "pooled_cache": pooled_cache.audit(),
        "encoder_trainable": False,
        "model_spec": model.model_spec(),
        "parameter_deltas": deltas,
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, output_root / "checkpoint.pt")
    report = {
        "status": "complete",
        "training_seed": training_seed,
        "projection_variant": model.projection_variant,
        "adaptive_basis_lr": model.adaptive_basis_lr,
        "pooled_cache": pooled_cache.audit(),
        "severity_bias_calibration": severity_bias_calibration,
        "severity_base_normalization": severity_base_normalization,
        **deltas,
        "history": history,
    }
    write_json(output_root / "refit_report.json", report)
    write_json(
        output_root / "report.json",
        {
            "status": "complete",
            "dataset": dataset,
            "fold_id": fold_id,
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "training_seed": training_seed,
            "projection_variant": model.projection_variant,
            "adaptive_basis_lr": model.adaptive_basis_lr,
            "best_epoch": int(best_epoch),
            "state_decision": STATE_DECISION,
            "severity_threshold": SEVERITY_THRESHOLD,
            "severity_bias_calibration": severity_bias_calibration,
            "severity_base_normalization": severity_base_normalization,
            "state_deviation_weight": STATE_DEVIATION_WEIGHT,
            "state_deviation_margin": STATE_DEVIATION_MARGIN,
            "severity_gradient_scope": "shared_kan_and_severity_head",
            "pooled_cache": pooled_cache.audit(),
            **deltas,
        },
    )
    return report


def _smoke(
    config: dict,
    dataset: str,
    fold_id: str,
    bank,
    pooled_cache: PooledFeatureCache,
    protocol: FoldProtocol,
    device: torch.device,
    output_root: Path,
) -> dict:
    training_seed = _training_seed(config)
    _seed_everything(training_seed)
    model, _ = _new_model(config, device)
    smoke_examples = tuple(
        example
        for example in protocol.source_examples
        if example.subject_id in set(protocol.source_train_subjects)
    )
    severity_base_normalization = _severity_base_normalization(
        model,
        bank,
        pooled_cache.features,
        pooled_cache.token_dynamics,
        smoke_examples,
    )
    before = {
        "encoder": _snapshot(model.encoder),
        "kan": _snapshot(model.shared_kan),
        "state_head": _snapshot(model.state_head),
        "severity_head": _snapshot(model.severity_head),
    }
    result = _train_epoch(
        model,
        _optimizer(model),
        torch.amp.GradScaler("cuda", enabled=device.type == "cuda"),
        bank,
        pooled_cache.features,
        pooled_cache.token_dynamics,
        training_seed,
        protocol.source_train_subjects,
        smoke_examples,
        1,
        device,
        max_steps=1,
    )
    deltas = {
        "encoder_delta_l2": _delta(model.encoder, before["encoder"]),
        "kan_delta_l2": _delta(model.shared_kan, before["kan"]),
        "state_head_delta_l2": _delta(model.state_head, before["state_head"]),
        "severity_head_delta_l2": _delta(model.severity_head, before["severity_head"]),
    }
    if deltas["encoder_delta_l2"] != 0.0 or min(
        deltas[name]
        for name in (
            "kan_delta_l2",
            "state_head_delta_l2",
            "severity_head_delta_l2",
        )
    ) <= 0.0:
        raise AssertionError(f"Smoke freeze/update audit failed: {deltas}")
    report = {
        "status": "passed",
        "dataset": dataset,
        "fold_id": fold_id,
        "training_seed": training_seed,
        "projection_variant": model.projection_variant,
        "adaptive_basis_lr": model.adaptive_basis_lr,
        "pooled_cache": pooled_cache.audit(),
        "severity_base_normalization": severity_base_normalization,
        **result,
        **deltas,
    }
    write_json(output_root / "smoke_report.json", report)
    return report


def _run_fold(
    config: dict,
    dataset: str,
    fold_id: str,
    bank,
    pooled_cache: PooledFeatureCache,
    protocol: FoldProtocol,
    device: torch.device,
    output_root: Path,
) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    best_epoch, selection = _select_epoch(
        config, dataset, fold_id, bank, pooled_cache, protocol, device, output_root
    )
    refit = _refit(
        config,
        dataset,
        fold_id,
        bank,
        pooled_cache,
        protocol,
        best_epoch,
        device,
        output_root,
    )
    return {"status": "complete", "selection": selection, "refit": refit}


def run_training(
    config: dict,
    dataset: str,
    fold: int | None,
    device_name: str,
    smoke: bool,
    output_root: Path,
) -> dict:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    dataset_data = load_training_dataset(config, dataset)
    cache_model, _ = _new_model(config, device)
    pooled_cache = load_or_build_pooled_cache(
        config, dataset, dataset_data.bank, device, cache_model
    )
    del cache_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"{dataset} pooled FEMBA cache hit={pooled_cache.cache_hit} "
        f"seconds={pooled_cache.load_or_build_seconds:.2f} "
        f"path={pooled_cache.metadata['cache_path']}",
        flush=True,
    )
    if smoke:
        fold_id = f"fold_{int(fold or 1)}"
        Path(output_root).mkdir(parents=True, exist_ok=True)
        return _smoke(
            config,
            dataset,
            fold_id,
            dataset_data.bank,
            pooled_cache,
            dataset_data.folds[fold_id],
            device,
            Path(output_root),
        )
    fold_ids = [f"fold_{int(fold)}"] if fold is not None else sorted(dataset_data.folds)
    reports = {}
    for fold_id in fold_ids:
        root = Path(output_root) if fold is not None else Path(output_root) / fold_id
        reports[fold_id] = _run_fold(
            config,
            dataset,
            fold_id,
            dataset_data.bank,
            pooled_cache,
            dataset_data.folds[fold_id],
            device,
            root,
        )
    return {
        "status": "complete",
        "dataset": dataset,
        "training_seed": _training_seed(config),
        "pooled_cache": pooled_cache.audit(),
        "folds": reports,
    }
