"""Locked nonlinear C0 comparison using the existing selection/refit engine."""
from pathlib import Path
import itertools

import torch

from ..config import DEFAULT_PROTOCOL_ROOT
from ..evaluation.anchor_pilot import metrics_from_rows, score
from ..evaluation.io import read_json, read_csv, write_json, sha256_file
from ..model.c0_heads import head_factory
from ..model.encoder import TemporalEncoder
from ..ssl_protocol import digest_json, validate_metadata, assert_empty
from . import ssl_fivefold as full
from .anchor_pilot import train_pilot
from .probe_data import sample_digest, epoch_order

SCHEMA = "femba_c0_head_comparison_v1"
HEADS = ("fractional_dog_polykan", "mlp")
VARIANTS = ("A3", "A4")
DATASETS = ("vrq", "city")
TASKS = ("state", "severity")
FOLDS = tuple(f"fold_{i}" for i in range(1, 6))
PROTOCOL_PATH = DEFAULT_PROTOCOL_ROOT / "femba_c0_head_comparison.json"
ARTIFACTS = ("selection/report.json", "selection/history.json", "selection/initialization.json",
             "refit/final.pt", "refit/report.json", "refit/history.json",
             "evaluation/report.json", "evaluation/predictions.csv")
IDENTITY_KEYS = ("checkpoint_schema", "training_seed", "condition", "task", "dataset", "variant",
                 "fold_id", "smoke", "protocol", "binding", "split", "train_samples_sha256",
                 "val_samples_sha256", "anchors", "anchors_sha256", "source_samples_sha256",
                 "test_samples_sha256", "outer_anchors", "outer_anchors_sha256", "official_pretrain_sha256",
                 "encoder_trainable", "environment")


def experiment_protocol(seed=2001):
    if seed != 2001:
        raise ValueError("This paired experiment is locked to seed=2001")
    locked = read_json(PROTOCOL_PATH)
    if locked["schema"] != SCHEMA:
        raise ValueError("Wrong head comparison protocol")
    return {**full.experiment_protocol(seed), **locked}


def protocol_binding(protocol, assets, protocol_root, baseline):
    result = full.protocol_binding(protocol, assets, protocol_root)
    result.update(protocol_file_sha256=sha256_file(PROTOCOL_PATH),
                  fivefold_base_file_sha256=sha256_file(full.PROTOCOL_PATH),
                  baseline_manifest_sha256=baseline["manifest_sha256"])
    script = Path(__file__).resolve().parents[3] / "scripts/run_femba_c0_head_comparison.py"
    result["entrypoint_sha256"] = sha256_file(script)
    return result


def scored_identity(rows):
    return [{"sample_id": r["sample_id"], "subject_id": r["subject_id"], "session": r["session"],
             "window_indices": [int(i) for i in str(r["window_indices"]).split(",")]} for r in rows]


def verify_artifacts(job, root, identity):
    report = read_json(job / "report.json")
    if not set(ARTIFACTS) <= set(report["artifacts"]):
        raise RuntimeError("Missing required fold artifacts")
    result, rows = full.verify_job(job, root, identity)
    selection = read_json(job / "selection/report.json")
    refit = read_json(job / "refit/report.json")
    evaluation = read_json(job / "evaluation/report.json")
    expected = {k: v for k, v in identity.items() if k != "environment"}
    for value in (selection, refit, evaluation):
        validate_metadata(value, expected)
    if selection["initialization"] != refit["initialization"] or (
        selection["initialization"] != evaluation["initialization"] or
        selection["initialization"] != report["initialization"]):
        raise RuntimeError("Fold initialization provenance mismatch")
    if refit["refit_steps"] != full.refit_steps(selection["best_global_step"], selection["steps_per_epoch"],
                                                refit["source_steps_per_epoch"]):
        raise RuntimeError("Refit budget differs from selected fraction")
    for phase in (selection, refit):
        if phase["reload_max_abs_error"] != 0:
            raise RuntimeError("Reload audit failed")
    if not refit["refit_reset_verified"] or selection["first_epoch_order_sha256"] != report["selection_order_sha256"] or (
        refit["first_epoch_order_sha256"] != report["refit_order_sha256"]):
        raise RuntimeError("Refit reset or shuffle audit mismatch")
    return result, rows


def verify_linear_baselines(root):
    """Read-only artifact verification; never rebuild identity with the new implementation hash."""
    root = Path(root)
    baseline, manifest, binding = {}, [], None
    reference_protocol = full.experiment_protocol()
    for task, variant, dataset, fold in itertools.product(TASKS, VARIANTS, DATASETS, FOLDS):
        job = root / "C0" / task / variant / dataset / fold
        report = read_json(job / "report.json")
        validate_metadata(report, {"checkpoint_schema": full.SCHEMA, "status": "complete", "smoke": False,
            "training_seed": 2001, "condition": "C0", "task": task, "variant": variant,
            "dataset": dataset, "fold_id": fold, "protocol": reference_protocol})
        identity = {k: report[k] for k in IDENTITY_KEYS}
        report, rows = verify_artifacts(job, root, identity)
        init = report["initialization"]
        if init["head_parameters"] != 526 or init["pretrain_load_info"]["loaded_keys"] != 83 or (
            init["pretrain_checkpoint_sha256"] != identity["official_pretrain_sha256"]) or any(
                init["pretrain_load_info"][k] for k in ("missing_keys", "unexpected_keys", "skipped_keys")):
            raise RuntimeError("Wrong linear baseline architecture or pretraining")
        if binding is None:
            binding = report["binding"]
        if report["binding"] != binding:
            raise RuntimeError("Mixed linear baseline protocols or implementations")
        train, val, source, test = (set(identity["split"][k]) for k in (
            "source_train_subjects", "source_val_subjects", "source_subjects", "test_subjects"))
        if train & val or source & test or train | val != source:
            raise RuntimeError("Baseline identity overlap")
        key = (task, variant, dataset, fold)
        baseline[key] = {"report": report, "rows": rows, "identity": identity}
        manifest.append({"job": "/".join(key), "report_sha256": sha256_file(job / "report.json"),
                         "artifacts": report["artifacts"]})
    return {"jobs": baseline, "manifest": manifest, "manifest_sha256": digest_json(manifest), "binding": binding}


def check_baseline_pair(identity, partitions, baseline):
    key = tuple(identity[k] for k in ("task", "variant", "dataset", "fold_id"))
    previous = baseline["jobs"][key]
    old = previous["identity"]
    intentional_changes = {"schema", "head", "anchor_count", "anchor_selection", "anchor_gradient",
                           "conditions", "condition_order", "limits"}
    for key, value in old["protocol"].items():
        if key not in intentional_changes and identity["protocol"].get(key) != value:
            raise RuntimeError(f"Training protocol differs from C0 baseline: {key}")
    for k in ("split", "train_samples_sha256", "val_samples_sha256", "source_samples_sha256",
              "test_samples_sha256", "anchors_sha256", "outer_anchors_sha256", "official_pretrain_sha256"):
        if identity[k] != old[k]:
            raise RuntimeError(f"C0 baseline pairing mismatch: {k}")
    for k in ("data_sha256", "bundle_sha256"):
        if identity["binding"][k] != old["binding"][k]:
            raise RuntimeError(f"C0 data binding mismatch: {k}")
    train, _, source, test, _, _ = partitions
    for name, examples in (("selection_order_sha256", train), ("refit_order_sha256", source)):
        order = digest_json([examples[int(i)].sample_id for i in epoch_order(len(examples), 2001, 1)])
        if order != previous["report"][name]:
            raise RuntimeError("Baseline target shuffle differs")
    if {e.sample_id: e.label for e in test} != {r["sample_id"]: int(r["y_true"]) for r in previous["rows"]}:
        raise RuntimeError("Baseline test labels or samples differ")


def verify_head_audit(job, identity, initial, baseline_initial=None):
    if initial["head_kind"] != identity["head"]:
        raise RuntimeError("Wrong head initialization")
    if baseline_initial is not None and initial["encoder_sha256"] != baseline_initial["encoder_sha256"]:
        raise RuntimeError("Encoder initialization differs from C0 baseline")
    history = read_json(job / "selection/history.json")["checks"]
    names = sorted({n for row in history for n in row["training"]["head_gradient_names"]})
    if identity["head"] == "fractional_dog_polykan" and not {
        "1.fractional_order_logit", "1.dog_mix_logits"} <= set(names):
        raise RuntimeError("Fractional/DoG mixing parameters never received nonzero gradients")
    return {"head_gradient_names": names, "baseline_encoder_initialization_verified": baseline_initial is not None}


def run_job(job, root, identity, bank, partitions, device, *, stage="all", resume=False,
            checkpoint_path=None, encoder_factory=TemporalEncoder, baseline_initial=None):
    if identity["checkpoint_schema"] != SCHEMA or identity["condition"] != "C0":
        raise ValueError("Only the locked C0 comparison schema is accepted")
    factory = head_factory(identity["head"])
    options = {"model_factory": factory, "checkpoint_schema": SCHEMA, "encoder_factory": encoder_factory}
    expected = {k: v for k, v in identity.items() if k != "environment"}
    if (job / "report.json").exists():
        if not resume and stage != "evaluate":
            raise FileExistsError(f"Completed fold exists: {job}")
        report, old_rows = verify_artifacts(job, root, identity)
        if stage == "evaluate":
            model, _ = full.restore_final(job / "refit/final.pt", root, identity, device, **options)
            rows, _ = score(model, bank, partitions[3], identity["task"], device,
                            identity["protocol"]["microbatch_size"][identity["task"]], partitions[5])
            if rows != old_rows:
                raise RuntimeError("Repeated evaluation differs from locked predictions")
        return report
    if resume and (job / "smoke_report.json").exists():
        report = read_json(job / "smoke_report.json")
        validate_metadata(report, expected)
        payload = full.final_payload(job / "refit/final.pt", root)
        validate_metadata(payload["identity"], expected)
        return report
    train, val, source, test, anchors, outer = partitions
    if stage == "evaluate":
        if identity["smoke"]:
            raise ValueError("Smoke cannot evaluate outer-test")
        marker = read_json(job / "trained.json")
        validate_metadata(marker, expected)
        for name, sha in marker["artifacts"].items():
            if sha256_file(job / name) != sha:
                raise RuntimeError("Trained artifact changed before evaluation")
        selection = read_json(job / "selection/report.json")
        refit = read_json(job / "refit/report.json")
    else:
        if not resume:
            assert_empty(job)
        selection = train_pilot(job / "selection", identity, bank, train, val, anchors, device,
            checkpoint_path=checkpoint_path, resume=resume, **options)
        refit = full.run_refit(job / "refit", root, identity, selection, bank, source, anchors, device,
            checkpoint_path=checkpoint_path, resume=resume, **options)
    audit = verify_head_audit(job, identity, selection["initialization"], baseline_initial)
    if identity["smoke"]:
        if selection["global_step"] < 2 or refit["refit_steps"] < 2:
            raise RuntimeError("Smoke requires two updates in both training phases")
        model, _ = full.restore_final(job / "refit/final.pt", root, identity, device, **options)
        _, refit_validation = score(model, bank, val, identity["task"], device,
                                    identity["protocol"]["microbatch_size"][identity["task"]], anchors)
        report = {**identity, "status": "smoke_passed", "selection": selection, "refit": refit,
                  "refit_source_val": refit_validation, "head_audit": audit, "outer_test_scored": False}
        write_json(job / "smoke_report.json", report)
        full.compact_intermediates(job, root, checkpoint_schema=SCHEMA)
        return report
    if stage == "train":
        report = {**identity, "status": "trained", "artifacts": {
            n: sha256_file(job / n) for n in ARTIFACTS if not n.startswith("evaluation/")}}
        write_json(job / "trained.json", report)
        return report
    evaluation = full.evaluate_outer(job, root, identity, bank, test, outer, device, **options)
    report = {**identity, "status": "complete", "initialization": selection["initialization"],
              "selection_best_step": selection["best_global_step"], "selection_best_epoch": selection["best_epoch"],
              "selection_validation": selection["best_validation"], "refit_steps": refit["refit_steps"],
              "refit_reset_verified": True, "scores": evaluation["scores"], "head_audit": audit,
              "selection_order_sha256": selection["first_epoch_order_sha256"],
              "refit_order_sha256": refit["first_epoch_order_sha256"],
              "artifacts": {n: sha256_file(job / n) for n in ARTIFACTS}}
    write_json(job / "report.json", report)
    verify_artifacts(job, root, identity)
    full.compact_intermediates(job, root, checkpoint_schema=SCHEMA)
    return report
