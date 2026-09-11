from pathlib import Path
import itertools
import os
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..anchor_protocol import SCHEMA, CONDITIONS
from ..model.anchor_probe import FEMBAAnchorProbe
from ..model.encoder import TemporalEncoder
from ..model.linear_probe import tensor_state_hash
from ..ssl_protocol import VARIANTS, DATASETS, TASKS, validate_metadata, assert_empty, digest_json
from ..training.anchor_data import batch_inputs
from .io import read_json, read_csv, write_json, write_csv, sha256_file
from .metrics import binary_metrics
from .ssl_ablation import _row_identity


def metrics_from_rows(rows):
    labels = [int(r["y_true"]) for r in rows]
    metrics = binary_metrics(rows) if set(labels) == {0, 1} else {
        "n_samples": len(rows), "accuracy": float(np.mean([int(r["y_pred"]) == y for r, y in zip(rows, labels)])),
        "balanced_accuracy": None, "AUROC": None}
    counts = {str(c): labels.count(c) for c in (0, 1)}
    return {"loss": F.binary_cross_entropy_with_logits(
        torch.tensor([float(r["logit"]) for r in rows], dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32)).item(), "metrics": metrics,
        "class_counts": counts, "positive_fraction": counts["1"] / len(rows),
        "majority_accuracy": max(counts.values()) / len(rows)}


@torch.no_grad()
def score(model, bank, examples, task, device, microbatch_size, anchors):
    model.eval()
    rows, encoded = [], 0
    for start in range(0, len(examples), microbatch_size):
        batch = examples[start:start + microbatch_size]
        inputs, count = batch_inputs(bank, batch, task, device, model.condition, anchors)
        encoded += count
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(*inputs).float()
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Non-finite pilot logits")
        for e, logit, probability in zip(batch, logits.cpu().tolist(), logits.sigmoid().cpu().tolist()):
            rows.append({"sample_id": e.sample_id, "subject_id": e.subject_id, "session": e.session,
                         "window_indices": ",".join(map(str, e.indices)), "logit": logit,
                         "score": probability, "threshold": 0.5, "y_pred": int(probability >= 0.5)})
    # Ground truth enters only after every prediction is computed.
    labels = {e.sample_id: e.label for e in examples}
    rows = [{**r, "y_true": labels[r["sample_id"]],
             "correct": int(r["y_pred"] == labels[r["sample_id"]])} for r in rows]
    return rows, {**metrics_from_rows(rows), "encoded_windows": encoded}


def load_verified(path, *, recover=False):
    path = Path(path)
    manifest_path = path.with_suffix(".sha256.json")
    valid = path.is_file() and manifest_path.is_file() and sha256_file(path) == read_json(manifest_path)["sha256"]
    if not valid and recover:
        previous = path.with_suffix(".previous.pt")
        if previous.is_file():
            sha = sha256_file(previous)
            manifests = (path.with_suffix(".previous.sha256.json"), manifest_path)
            if any(p.is_file() and read_json(p)["sha256"] == sha for p in manifests):
                if path.exists():
                    path.rename(path.with_suffix(f".interrupted-{time.time_ns()}.pt"))
                os.replace(previous, path)
                write_json(manifest_path, {"sha256": sha})
                valid = True
    if not valid:
        raise RuntimeError(f"Checkpoint file integrity mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def restore_checkpoint(path, expected, device, encoder_factory=TemporalEncoder, *, expected_schema=SCHEMA, model_factory=None):
    return restore_payload(load_verified(path), expected, device, encoder_factory,
                           expected_schema=expected_schema, model_factory=model_factory)


def restore_payload(payload, expected, device, encoder_factory=TemporalEncoder, *, expected_schema=SCHEMA, model_factory=None):
    validate_metadata(payload["identity"], {k: v for k, v in expected.items() if k != "environment"})
    if payload["identity"]["checkpoint_schema"] != expected_schema:
        raise ValueError("Wrong pilot checkpoint schema")
    identity, initial = payload["identity"], payload["initialization"]
    spec = VARIANTS[identity["variant"]]
    if spec["pretrained"]:
        info = initial["pretrain_load_info"]
        if initial["pretrain_checkpoint_sha256"] != identity["official_pretrain_sha256"] or not info or (
            info["loaded_keys"] != 83 or any(info[k] for k in ("missing_keys", "unexpected_keys", "skipped_keys"))
        ):
            raise RuntimeError("Wrong pretrained provenance")
    elif initial["pretrain_checkpoint_sha256"] is not None or initial["pretrain_load_info"] is not None:
        raise RuntimeError("Random group has pretrained provenance")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(identity["training_seed"])
        factory = FEMBAAnchorProbe if model_factory is None else model_factory
        model = factory(encoder_factory(), encoder_trainable=spec["encoder_trainable"],
                                seed=identity["training_seed"], condition=identity["condition"])
    if model_factory is not None:
        validate_metadata(initial, model.head_initialization())
    if {n: list(p.shape) for n, p in model.named_parameters()} != initial["parameter_shapes"]:
        raise RuntimeError("Parameter shapes changed")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if tensor_state_hash(model.state_dict()) != payload["model_sha256"]:
        raise RuntimeError("Model hash mismatch")
    if not spec["encoder_trainable"] and tensor_state_hash(model.encoder.state_dict()) != initial["encoder_sha256"]:
        raise RuntimeError("Frozen encoder changed")
    return model.to(device).eval(), payload


def evaluate(checkpoint, output, identity, bank, examples, anchors, device, encoder_factory=TemporalEncoder):
    assert_empty(output)
    if digest_json(anchors) != identity["anchors_sha256"]:
        raise ValueError("Validation anchor hash mismatch")
    if set(e.subject_id for e in examples) != set(identity["split"]["source_val_subjects"]):
        raise ValueError("Pilot evaluation accepts source-val subjects only")
    from ..training.probe_data import sample_digest
    if sample_digest(examples) != identity["val_samples_sha256"]:
        raise ValueError("Validation sample hash mismatch")
    model, payload = restore_checkpoint(checkpoint, identity, device, encoder_factory)
    if payload["role"] != "best" or payload["global_step"] < 1:
        raise ValueError("Evaluation requires a selected best checkpoint")
    rows, validation = score(model, bank, examples, identity["task"], device,
                             identity["protocol"]["microbatch_size"][identity["task"]], anchors)
    if validation != payload["validation"]:
        raise AssertionError("Reloaded best checkpoint validation changed")
    write_csv(output / "predictions.csv", rows)
    report = {**identity, "status": "smoke_passed" if identity["smoke"] else "complete",
              "evaluation_partition": "source_val", "best_global_step": payload["global_step"],
              "best_epoch": payload["epoch"], "validation": validation,
              "initialization": payload["initialization"], "reload_max_abs_error": 0.0,
              "checkpoint_sha256": sha256_file(checkpoint),
              "predictions_sha256": sha256_file(output / "predictions.csv")}
    write_json(output / "report.json", report)
    return report


def verify_result(job, identity):
    job = Path(job)
    train, report = read_json(job / "training/report.json"), read_json(job / "evaluation/report.json")
    for value in (train, report):
        validate_metadata(value, {k: v for k, v in identity.items() if k != "environment"})
        if value["status"] != ("smoke_passed" if identity["smoke"] else "complete"):
            raise RuntimeError("Incomplete pilot result")
    if report["evaluation_partition"] != "source_val":
        raise RuntimeError("Not validation-only results")
    if sha256_file(job / "evaluation/predictions.csv") != report["predictions_sha256"] or (
        sha256_file(job / "training/best.pt") != report["checkpoint_sha256"]
    ):
        raise RuntimeError("Result artifact hash mismatch")
    rows = read_csv(job / "evaluation/predictions.csv")
    sample_records = [{"sample_id": r["sample_id"], "subject_id": r["subject_id"],
                       "session": r["session"], "window_indices": [int(i) for i in str(r["window_indices"]).split(",")]}
                      for r in rows]
    if digest_json(sample_records) != identity["val_samples_sha256"]:
        raise RuntimeError("Validation scored sample hash mismatch")
    rescored = metrics_from_rows(rows)
    if any(rescored[k] != report["validation"][k] for k in rescored):
        raise RuntimeError("Stored metrics do not reproduce")
    if set(r["subject_id"] for r in rows) != set(identity["split"]["source_val_subjects"]):
        raise RuntimeError("Scoring partition mismatch")
    if train["best_validation"] != report["validation"] or train["initialization"] != report["initialization"]:
        raise RuntimeError("Training and evaluation disagree")
    if report["reload_max_abs_error"] != 0 or train["reload_max_abs_error"] != 0:
        raise RuntimeError("Reload audit failed")
    for filename in ("best.pt", "last.pt", "boundary.pt", "history.json", "initialization.json"):
        checkpoint = job / "training" / filename
        if sha256_file(checkpoint) != train["artifacts"][filename]:
            raise RuntimeError("Training checkpoint integrity mismatch")
    payload = load_verified(job / "training/best.pt")
    validate_metadata(payload["identity"], {k: v for k, v in identity.items() if k != "environment"})
    if payload["role"] != "best" or payload["global_step"] != report["best_global_step"] or (
        payload["initialization"] != report["initialization"] or
        tensor_state_hash(payload["model_state_dict"]) != payload["model_sha256"]
    ):
        raise RuntimeError("Best checkpoint provenance mismatch")
    return train, report, rows


def contrasts(values):
    # values[(condition, variant)] is one metric. All differences are percentage points.
    if any(v is None for v in values.values()):
        return None
    gain = {v: 100 * (values["C1", v] - values["C0", v]) for v in VARIANTS}
    pf = {c: 100 * (values[c, "A3"] - values[c, "A1"]) for c in CONDITIONS}
    pt = {c: 100 * (values[c, "A4"] - values[c, "A2"]) for c in CONDITIONS}
    interaction = {"frozen": pf["C1"] - pf["C0"], "trainable": pt["C1"] - pt["C0"]}
    return {"anchor_gain_pp": gain, "frozen_pretraining_gain_pp": pf,
            "trainable_pretraining_gain_pp": pt, "interaction_pp": interaction,
            "finetune_minus_frozen_pp": {c: 100 * (values[c, "A4"] - values[c, "A3"]) for c in CONDITIONS},
            "random_supervised_gain_pp": {c: 100 * (values[c, "A2"] - values[c, "A1"]) for c in CONDITIONS},
            "improvement_and_amplification": {"frozen": gain["A3"] > 0 and interaction["frozen"] > 0,
                                               "trainable": gain["A4"] > 0 and interaction["trainable"] > 0},
            "pretrained_beats_random_with_anchors": {"frozen": pf["C1"] > 0, "trainable": pt["C1"] > 0}}


def summarize(root, identities):
    root = Path(root)
    expected = set(itertools.product(TASKS, CONDITIONS, VARIANTS, DATASETS))
    actual = {(i["task"], i["condition"], i["variant"], i["dataset"]) for i in identities}
    if actual != expected or len(identities) != 48 or any(
        i["smoke"] or i["fold_id"] != "fold_1" or i["checkpoint_schema"] != SCHEMA for i in identities
    ):
        raise ValueError("Complete pilot summary requires all 48 non-smoke conditions")
    assert_empty(root / "summary")
    table, reports, paired = [], {}, {}
    for identity in identities:
        task, c, v, d = (identity[k] for k in ("task", "condition", "variant", "dataset"))
        job = root / task / c / v / d / "fold_1"
        train, report, rows = verify_result(job, identity)
        initial = report["initialization"]
        checks = {("samples", task, d): _row_identity(rows),
                  ("anchors", d): identity["anchors_sha256"],
                  ("train", task, d): identity["train_samples_sha256"],
                  ("shuffle", task, d): train["first_epoch_order_sha256"],
                  ("head",): initial["head_sha256"], ("shapes",): initial["parameter_shapes"],
                  ("encoder", VARIANTS[v]["pretrained"]): initial["encoder_sha256"],
                  ("protocol",): identity["protocol"], ("binding",): identity["binding"]}
        for key, value in checks.items():
            if paired.setdefault(key, value) != value:
                raise RuntimeError(f"Paired audit mismatch: {key}")
        scores = report["validation"]
        metrics = scores["metrics"]
        reports[task, d, c, v] = metrics
        table.append({"task": task, "dataset": d, "condition": c, "variant": v,
                      **{k: metrics[k] for k in ("accuracy", "balanced_accuracy", "AUROC")},
                      "validation_BCE": scores["loss"], "majority_accuracy": scores["majority_accuracy"],
                      "positive_fraction": scores["positive_fraction"], "best_step": report["best_global_step"],
                      "steps": train["global_step"], "training_seconds": train["training_seconds"],
                      "peak_cuda_bytes": train["peak_cuda_bytes"]})
    comparisons, macros = {}, []
    for task in TASKS:
        comparisons[task] = {}
        for dataset in (*DATASETS, "dataset_macro"):
            by_metric = {}
            for metric in ("accuracy", "balanced_accuracy", "AUROC"):
                values = {}
                for c, v in itertools.product(CONDITIONS, VARIANTS):
                    data = [reports[task, d, c, v][metric] for d in DATASETS] if dataset == "dataset_macro" else [reports[task, dataset, c, v][metric]]
                    values[c, v] = None if any(x is None for x in data) else float(np.mean(data))
                    if dataset == "dataset_macro":
                        macros.append({"task": task, "condition": c, "variant": v,
                                       "metric": metric, "value": values[c, v]})
                by_metric[metric] = contrasts(values)
            comparisons[task][dataset] = by_metric
    result = {"status": "complete", "pilot_completed": True, "full_fivefold_completed": False,
              "jobs": 48, "protocol": identities[0]["protocol"], "binding": identities[0]["binding"],
              "metrics": table, "dataset_macros": macros, "comparisons": comparisons,
              "limits": identities[0]["protocol"]["limits"]}
    write_csv(root / "summary/metrics.csv", table)
    write_csv(root / "summary/dataset_macros.csv", macros)
    write_json(root / "summary/aggregate_report.json", result)
    return result
