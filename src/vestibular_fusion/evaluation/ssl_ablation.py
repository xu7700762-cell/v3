from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..model.linear_probe import FEMBALinearProbe, tensor_state_hash
from ..model.encoder import TemporalEncoder
from ..ssl_protocol import (SCHEMA, VARIANTS, FOLDS, DATASETS, assert_empty,
                            validate_metadata)
from ..training.probe_data import input_tensor, sample_digest
from .io import read_csv, read_json, sha256_file, write_csv, write_json
from .metrics import binary_metrics


@torch.no_grad()
def predict(model, bank, examples, task: str, device, batch_size: int) -> list[dict]:
    """Score without reading example labels; attach labels only after all scoring."""
    model.eval()
    rows = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        windows = input_tensor(bank, batch, task, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(windows).float()
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Non-finite probe prediction")
        scores = logits.sigmoid().cpu().tolist()
        for e, logit, score in zip(batch, logits.cpu().tolist(), scores):
            rows.append({"sample_id": e.sample_id, "subject_id": e.subject_id,
                         "session": e.session, "window_indices": ",".join(map(str, e.indices)),
                         "logit": logit, "score": score, "threshold": 0.5,
                         "y_pred": int(score >= 0.5)})
    return rows


def score(model, bank, examples, task: str, device, batch_size: int):
    rows = predict(model, bank, examples, task, device, batch_size)
    labels = {e.sample_id: e.label for e in examples}
    rows = [{**r, "y_true": labels[r["sample_id"]],
             "correct": int(r["y_pred"] == labels[r["sample_id"]])} for r in rows]
    loss = F.binary_cross_entropy_with_logits(
        torch.tensor([r["logit"] for r in rows], dtype=torch.float32),
        torch.tensor([r["y_true"] for r in rows], dtype=torch.float32),
    ).item()
    # AUROC/BACC require both classes; single-class validation still has valid BCE.
    metrics = binary_metrics(rows) if set(labels.values()) == {0, 1} else {
        "n_samples": len(rows), "accuracy": float(np.mean([r["correct"] for r in rows])),
        "balanced_accuracy": None, "AUROC": None,
    }
    return rows, {"loss": loss, "metrics": metrics}


def restore_checkpoint(path: Path, expected: dict, device, encoder_factory=TemporalEncoder):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    # Hardware/kernel descriptions are provenance, not protocol compatibility keys.
    validate_metadata(payload, {k: v for k, v in expected.items() if k != "environment"})
    variant = VARIANTS[payload["variant"]]
    validate_metadata(payload, {"checkpoint_schema": SCHEMA,
                                "encoder_trainable": variant["encoder_trainable"]})
    initial = payload["initialization"]
    if variant["pretrained"]:
        if initial["pretrain_checkpoint_sha256"] != payload["official_pretrain_sha256"]:
            raise RuntimeError("Wrong pretrained checkpoint provenance")
        info = initial["pretrain_load_info"]
        if not info or info["loaded_keys"] != 83 or any(info[k] for k in (
            "missing_keys", "unexpected_keys", "skipped_keys"
        )):
            raise RuntimeError("Incomplete pretrained checkpoint provenance")
    elif initial["pretrain_checkpoint_sha256"] is not None or initial["pretrain_load_info"] is not None:
        raise RuntimeError("Random initialization cannot have pretrained provenance")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(payload["training_seed"])
        encoder = encoder_factory()
    model = FEMBALinearProbe(encoder, encoder_trainable=variant["encoder_trainable"],
                             seed=payload["training_seed"])
    if {n: list(p.shape) for n, p in model.named_parameters()} != initial["parameter_shapes"]:
        raise RuntimeError("Probe parameter shapes changed")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if tensor_state_hash(model.state_dict()) != payload["final_model_sha256"]:
        raise RuntimeError("Saved model state hash mismatch")
    if not variant["encoder_trainable"] and (
        tensor_state_hash(model.encoder.state_dict()) != initial["encoder_sha256"]
    ):
        raise RuntimeError("Frozen encoder changed in saved checkpoint")
    return model.to(device).eval(), payload


def evaluate_fold(checkpoint: Path, output: Path, expected: dict, bank, examples, device,
                  *, encoder_factory=TemporalEncoder) -> dict:
    if expected.get("smoke"):
        raise ValueError("Smoke checkpoints cannot produce complete outer-test evaluations")
    assert_empty(output)
    model, payload = restore_checkpoint(checkpoint, expected, device, encoder_factory=encoder_factory)
    if payload["smoke"] or not payload["refit_reset_verified"] or not (
        payload["protocol"]["min_epochs"] <= payload["best_epoch"] <= payload["protocol"]["max_epochs"]
    ):
        raise RuntimeError("Checkpoint did not complete selection and source refit")
    task = payload["task"]
    if payload["test_samples_sha256"] != sample_digest(examples):
        raise RuntimeError("Outer-test sample set changed")
    rows, scores = score(model, bank, examples, task, device, payload["protocol"]["batch_size"][task])
    report = {**expected, "status": "complete", "metrics": scores["metrics"],
              "loss": scores["loss"], "test_samples_sha256": sample_digest(examples),
              "checkpoint_sha256": sha256_file(checkpoint),
              "initialization": payload["initialization"], "audit": payload["audit"],
              "training_environment": payload.get("environment"),
              "best_epoch": payload["best_epoch"], "limits": payload["protocol"]["limits"]}
    write_csv(output / "predictions.csv", rows)
    report["predictions_sha256"] = sha256_file(output / "predictions.csv")
    write_json(output / "report.json", report)
    return report


def _row_identity(rows):
    identities = sorted((str(r["sample_id"]), str(r["subject_id"]), str(r["session"]),
                         str(r["window_indices"]), int(r["y_true"])) for r in rows)
    if len({r[0] for r in identities}) != len(identities):
        raise RuntimeError("Duplicate scored samples")
    return identities


def summarize(root: Path, protocol: dict, binding: dict, official_sha: str,
              tasks: list[str], expected_folds: dict) -> dict:
    """A complete summary always requires A1..A4, three datasets and all five folds."""
    destination = root / "summary"
    assert_empty(destination)
    groups, table = {}, []
    paired = {}
    for task in tasks:
        groups[task] = {}
        for variant in VARIANTS:
            groups[task][variant] = {}
            for dataset in DATASETS:
                all_rows, fold_metrics = [], []
                for fold_id in FOLDS:
                    folder = root / task / variant / dataset / fold_id / "evaluation"
                    report = read_json(folder / "report.json")
                    expected = {"checkpoint_schema": SCHEMA, "status": "complete", "smoke": False,
                                "variant": variant, "task": task, "dataset": dataset, "fold_id": fold_id,
                                "training_seed": protocol["training_seed"], "binding": binding,
                                "protocol": protocol, "official_pretrain_sha256": official_sha,
                                "encoder_trainable": VARIANTS[variant]["encoder_trainable"],
                                "split": expected_folds[dataset][fold_id]}
                    validate_metadata(report, expected)
                    csv_path = folder / "predictions.csv"
                    if sha256_file(csv_path) != report["predictions_sha256"]:
                        raise RuntimeError("Prediction CSV hash mismatch")
                    checkpoint = folder.parent / "training" / "checkpoint.pt"
                    if sha256_file(checkpoint) != report["checkpoint_sha256"]:
                        raise RuntimeError("Evaluation checkpoint hash mismatch")
                    rows = read_csv(csv_path)
                    identities = _row_identity(rows)
                    if {r["subject_id"] for r in rows} != set(expected["split"]["test_subjects"]):
                        raise RuntimeError("Scored test subjects do not match the locked split")
                    metrics = binary_metrics(rows)
                    if metrics != report["metrics"]:
                        raise RuntimeError("Reported metrics do not reproduce from predictions")
                    key = (task, dataset, fold_id)
                    init = report["initialization"]
                    common = {"samples": identities, "head": init["head_sha256"],
                              "shapes": init["parameter_shapes"]}
                    pair = paired.setdefault(key, {"common": common, "encoders": {}, "orders": {}})
                    if pair["common"] != common:
                        raise RuntimeError("Four-way sample/head/architecture pairing failed")
                    family = "pretrained" if VARIANTS[variant]["pretrained"] else "random"
                    previous = pair["encoders"].setdefault(family, init["encoder_sha256"])
                    if previous != init["encoder_sha256"]:
                        raise RuntimeError("Paired encoder initialization differs")
                    # Compare the common prefix even if validation selected different epoch counts.
                    for epoch, order in report["audit"]["epoch_order_sha256"].items():
                        if pair["orders"].setdefault(epoch, order) != order:
                            raise RuntimeError("Paired training sampling differs")
                    fold_metrics.append(metrics)
                    all_rows.extend({**r, "fold_id": fold_id} for r in rows)
                _row_identity(all_rows)
                combined = binary_metrics(all_rows)
                names = ("accuracy", "balanced_accuracy", "AUROC")
                stats = {n: {"mean": float(np.mean([m[n] for m in fold_metrics])),
                             "std": float(np.std([m[n] for m in fold_metrics], ddof=1))}
                         for n in names}
                groups[task][variant][dataset] = {"pooled_metrics": combined,
                                                 "fold_mean_std": stats, "fold_metrics": fold_metrics}
                table.append({"task": task, "variant": variant, "dataset": dataset,
                              **{n: combined[n] for n in names}})
    contrasts = {}
    for task in tasks:
        for variant in VARIANTS:
            groups[task][variant]["macro"] = {
                n: float(np.mean([groups[task][variant][d]["pooled_metrics"][n] for d in DATASETS]))
                for n in ("accuracy", "balanced_accuracy", "AUROC")
            }
        contrasts[task] = {}
        for left, right in (("A3", "A1"), ("A4", "A2"), ("A4", "A3"), ("A2", "A1")):
            contrasts[task][f"{left}-{right}"] = {}
            for dataset in (*DATASETS, "macro"):
                a = groups[task][left][dataset]
                b = groups[task][right][dataset]
                if dataset != "macro":
                    a, b = a["pooled_metrics"], b["pooled_metrics"]
                contrasts[task][f"{left}-{right}"][dataset] = {
                    name: 100.0 * (a[name] - b[name])
                    for name in ("accuracy", "balanced_accuracy", "AUROC")
                }
    report = {"status": "complete", "checkpoint_schema": SCHEMA, "smoke": False,
              "tasks": tasks, "protocol": protocol, "binding": binding,
              "groups": groups, "contrasts_percentage_points": contrasts,
              "limits": protocol["limits"]}
    write_csv(destination / "metrics.csv", table)
    write_json(destination / "aggregate_report.json", report)
    return report
