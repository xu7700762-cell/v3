"""Full source selection/refit/outer-test workflow, independent of v27."""
from pathlib import Path
import itertools
import math
import time

import numpy as np
import torch

from ..anchor_protocol import experiment_protocol as pilot_protocol, protocol_binding as pilot_binding
from ..config import DEFAULT_PROTOCOL_ROOT
from ..data.anchors import anchor_indices
from ..evaluation.anchor_pilot import (load_verified, restore_payload, score, metrics_from_rows, contrasts)
from ..evaluation.io import read_json, read_csv, write_json, write_csv, sha256_file
from ..evaluation.ssl_ablation import _row_identity
from ..model.anchor_probe import build_anchor_probe
from ..model.encoder import TemporalEncoder
from ..model.linear_probe import tensor_state_hash
from ..ssl_protocol import VARIANTS, TASKS, DATASETS, split_identity, digest_json, validate_metadata, assert_empty
from .anchor_data import pilot_partitions
from .anchor_pilot import train_pilot, optimizer_step, save_checkpoint, rng_state, restore_rng
from .probe_data import make_examples, sample_digest, class_weight, epoch_order
from .ssl_ablation import seed_runtime, snapshot, optimizer_for, update_audit

SCHEMA = "femba_ssl_fivefold_v3"
PROTOCOL_PATH = DEFAULT_PROTOCOL_ROOT / "femba_ssl_fivefold.json"


def experiment_protocol(seed=2001):
    return {**pilot_protocol(seed), **read_json(PROTOCOL_PATH)}


def protocol_binding(protocol, assets, protocol_root):
    value = pilot_binding(protocol, assets, protocol_root)
    value["selection_base_file_sha256"] = value.pop("protocol_file_sha256")
    value["protocol_file_sha256"] = sha256_file(PROTOCOL_PATH)
    package = Path(__file__).resolve().parents[1]
    source_files = sorted(package.rglob("*.py"))
    value["implementation_sha256"] = digest_json(
        [(p.relative_to(package).as_posix(), sha256_file(p)) for p in source_files] +
        [("scripts/run_femba_ssl_fivefold.py", sha256_file(package.parents[1] / "scripts/run_femba_ssl_fivefold.py"))])
    return value


def full_partitions(bank, fold, task):
    train, val, anchors, samples = pilot_partitions(bank, fold, task)
    source = make_examples(bank, fold, task, fold.source_subjects)
    test = make_examples(bank, fold, task, fold.test_subjects)
    refs = {}
    for e in fold.test_examples:
        if refs.setdefault(e.subject_id, e.reference_session) != e.reference_session:
            raise ValueError("Conflicting outer reference sessions")
    if set(refs) != set(fold.test_subjects):
        raise ValueError("Missing outer reference session")
    outer_anchors = {s: {"reference_session": ref, "indices": [int(i) for i in anchor_indices(bank.records[s], ref)]}
                     for s, ref in sorted(refs.items())}
    if set(anchors) & set(outer_anchors):
        raise ValueError("Source and outer-test anchors overlap")
    return train, val, source, test, anchors, outer_anchors, {
        **samples, "source_samples_sha256": sample_digest(source), "test_samples_sha256": sample_digest(test),
        "outer_anchors": outer_anchors, "outer_anchors_sha256": digest_json(outer_anchors)}


def refit_steps(best_step, selection_n, source_n):
    if min(best_step, selection_n, source_n) < 1:
        raise ValueError("Invalid refit budget")
    return (best_step * source_n + selection_n - 1) // selection_n


def store_final(path, payload, root):
    """Deduplicate only frozen, byte-identical encoder tensors; keep full precision."""
    value = dict(payload)
    if not VARIANTS[payload["identity"]["variant"]]["encoder_trainable"]:
        state = payload["model_state_dict"]
        encoder = {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}
        digest = tensor_state_hash(encoder)
        if digest != payload["initialization"]["encoder_sha256"]:
            raise AssertionError("Frozen encoder changed before shared storage")
        relative = Path("shared_encoders") / f"{digest}.pt"
        shared = root / relative
        shared.parent.mkdir(parents=True, exist_ok=True)
        if shared.exists():
            existing = load_verified(shared)
            if tensor_state_hash(existing["encoder_state_dict"]) != digest:
                raise RuntimeError("Shared encoder integrity mismatch")
        else:
            save_checkpoint(shared, {"encoder_state_dict": encoder, "encoder_sha256": digest})
        value["model_state_dict"] = {k: v for k, v in state.items() if k.startswith("head.")}
        value["encoder_reference"] = {"path": relative.as_posix(), "file_sha256": sha256_file(shared),
                                      "encoder_sha256": digest}
    save_checkpoint(path, value)


def final_payload(path, root):
    payload = load_verified(path)
    reference = payload.get("encoder_reference")
    if reference is not None:
        if VARIANTS[payload["identity"]["variant"]]["encoder_trainable"]:
            raise RuntimeError("Trainable encoder cannot use shared frozen storage")
        shared = (root / reference["path"]).resolve()
        shared.relative_to((root / "shared_encoders").resolve())
        if sha256_file(shared) != reference["file_sha256"]:
            raise RuntimeError("Shared encoder file hash mismatch")
        saved = load_verified(shared)
        if tensor_state_hash(saved["encoder_state_dict"]) != reference["encoder_sha256"] or (
            reference["encoder_sha256"] != payload["initialization"]["encoder_sha256"]
        ):
            raise RuntimeError("Shared encoder identity mismatch")
        payload["model_state_dict"] = {**{"encoder." + k: v for k, v in saved["encoder_state_dict"].items()},
                                       **payload["model_state_dict"]}
    if tensor_state_hash(payload["model_state_dict"]) != payload["model_sha256"]:
        raise RuntimeError("Final model hash mismatch")
    return payload


def restore_final(path, root, identity, device, encoder_factory=TemporalEncoder, *,
                  checkpoint_schema=SCHEMA, model_factory=None):
    payload = final_payload(path, root)
    if payload["role"] != "source_refit" or not payload["refit_reset_verified"]:
        raise ValueError("Outer evaluation requires a freshly initialized source refit")
    return restore_payload(payload, identity, device, encoder_factory,
                           expected_schema=checkpoint_schema, model_factory=model_factory)


def run_refit(output, root, identity, selection, bank, source, anchors, device, *, checkpoint_path=None,
              resume=False, encoder_factory=TemporalEncoder, step_callback=None,
              checkpoint_schema=SCHEMA, model_factory=None):
    if identity["checkpoint_schema"] != checkpoint_schema:
        raise ValueError("Refit checkpoint schema mismatch")
    output.mkdir(parents=True, exist_ok=True)
    expected = {k: v for k, v in identity.items() if k != "environment"}
    if set(e.subject_id for e in source) != set(identity["split"]["source_subjects"]) or (
        sample_digest(source) != identity["source_samples_sha256"] or digest_json(anchors) != identity["anchors_sha256"]
    ):
        raise ValueError("Source refit partition mismatch")
    if resume and (output / "report.json").exists():
        report = read_json(output / "report.json")
        validate_metadata(report, expected)
        if sha256_file(output / "final.pt") != report["final_checkpoint_sha256"]:
            raise RuntimeError("Refit checkpoint changed")
        return report
    if not resume:
        assert_empty(output)
    seed_runtime(identity["training_seed"])
    model, initial = build_anchor_probe(identity["variant"], identity["training_seed"], device,
        condition=identity["condition"], checkpoint_path=checkpoint_path,
        expected_sha256=identity["official_pretrain_sha256"], encoder_factory=encoder_factory,
        model_factory=model_factory)
    if initial != selection["initialization"]:
        raise AssertionError("Refit did not restore original encoder and head initialization")
    before, optimizer = snapshot(model), optimizer_for(model, identity["protocol"])
    p, task = identity["protocol"], identity["task"]
    source_n = math.ceil(len(source) / p["batch_size"][task])
    target = refit_steps(selection["best_global_step"], selection["steps_per_epoch"], source_n)
    if identity["smoke"]:
        target = max(2, target)
    weight = class_weight(source)
    step, epoch, history, received = 0, 0, [], set()
    boundary_path = output / "boundary.pt"

    def pack(role):
        state = snapshot(model)
        return {"identity": identity, "initialization": initial, "role": role,
                "model_state_dict": state, "model_sha256": tensor_state_hash(state),
                "global_step": step, "epoch": epoch, "target_steps": target,
                "refit_reset_verified": True}

    def boundary():
        save_checkpoint(boundary_path, {**pack("refit_boundary"), "optimizer": optimizer.state_dict(),
                        "rng": rng_state(), "history": history, "received": sorted(received),
                        "branch_gradients": model.branch_gradients})

    if resume and (boundary_path.exists() or boundary_path.with_suffix(".previous.pt").exists()):
        state = load_verified(boundary_path, recover=True)
        validate_metadata(state["identity"], expected)
        if state["initialization"] != initial or state["target_steps"] != target or state["global_step"] % source_n:
            raise RuntimeError("Refit boundary budget/initialization mismatch")
        model.load_state_dict(state["model_state_dict"], strict=True)
        if tensor_state_hash(model.state_dict()) != state["model_sha256"]:
            raise RuntimeError("Refit boundary model hash mismatch")
        optimizer.load_state_dict(state["optimizer"])
        step, epoch, history = state["global_step"], state["epoch"], state["history"]
        received, model.branch_gradients = set(state["received"]), state["branch_gradients"]
        restore_rng(state["rng"])
        del state
    else:
        if resume and any(output.iterdir()):
            raise RuntimeError("Refit has files but no committed boundary")
        boundary()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    while step < target:
        epoch += 1
        order = epoch_order(len(source), identity["training_seed"], epoch)
        for start in range(0, len(order), p["batch_size"][task]):
            batch = [source[int(i)] for i in order[start:start + p["batch_size"][task]]]
            row = optimizer_step(model, optimizer, bank, batch, task, device, p, anchors, weight)
            received.update(row["encoder_gradient_names"])
            step += 1
            history.append({"epoch": epoch, "global_step": step, **row})
            if step_callback is not None:
                step_callback(step)
            if step % source_n == 0:
                boundary()
            if step >= target:
                break
        write_json(output / "history.json", {"steps": history})
        print(f"REFIT {identity['condition']}/{task}/{identity['variant']}/{identity['dataset']}/{identity['fold_id']} "
              f"epoch={epoch} step={step}/{target}", flush=True)
    audit = update_audit(model, before)
    if model.encoder_trainable and received != set(dict(model.encoder.named_parameters())):
        raise AssertionError("Refit encoder gradient coverage failed")
    audit.update(encoder_gradient_names=sorted(received), branch_gradients=dict(model.branch_gradients))
    if model.encoder_trainable and (not model.branch_gradients["target"] or
        (identity["condition"] == "C1" and not model.branch_gradients["anchor"])):
        raise AssertionError("Refit encoder branch gradients missing")
    payload = {**pack("source_refit"), "audit": audit, "source_pos_weight": weight,
               "selection_best_step": selection["best_global_step"],
               "selection_steps_per_epoch": selection["steps_per_epoch"], "source_steps_per_epoch": source_n}
    path = output / "final.pt"
    store_final(path, payload, root)
    examples = source[:p["microbatch_size"][task]]
    first, _ = score(model, bank, examples, task, device, p["microbatch_size"][task], anchors)
    del model, optimizer, before, payload
    restored, _ = restore_final(path, root, identity, device, encoder_factory,
                                checkpoint_schema=checkpoint_schema, model_factory=model_factory)
    second, _ = score(restored, bank, examples, task, device, p["microbatch_size"][task], anchors)
    error = max(abs(a["logit"] - b["logit"]) for a, b in zip(first, second))
    if error != 0:
        raise AssertionError("Refit checkpoint reload changed logits")
    report = {**identity, "status": "complete", "initialization": initial, "audit": audit,
              "refit_reset_verified": True, "refit_steps": target, "source_steps_per_epoch": source_n,
              "selection_best_step": selection["best_global_step"], "source_pos_weight": weight,
              "refit_wall_seconds": time.perf_counter() - started, "reload_max_abs_error": error,
              "training_encoded_windows": sum(r["encoded_windows"] for r in history),
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
              "first_epoch_order_sha256": digest_json([source[int(i)].sample_id for i in epoch_order(len(source), identity["training_seed"], 1)]),
              "final_checkpoint_sha256": sha256_file(path)}
    write_json(output / "report.json", report)
    return report


def evaluate_outer(job, root, identity, bank, test, outer_anchors, device, encoder_factory=TemporalEncoder, *,
                   checkpoint_schema=SCHEMA, model_factory=None):
    if identity["smoke"]:
        raise ValueError("Smoke cannot score outer-test")
    if set(e.subject_id for e in test) != set(identity["split"]["test_subjects"]) or (
        sample_digest(test) != identity["test_samples_sha256"] or digest_json(outer_anchors) != identity["outer_anchors_sha256"]
    ):
        raise ValueError("Outer-test partition mismatch")
    model, payload = restore_final(job / "refit/final.pt", root, identity, device, encoder_factory,
                                   checkpoint_schema=checkpoint_schema, model_factory=model_factory)
    # Labels are attached by score only after all outer predictions are computed.
    rows, result = score(model, bank, test, identity["task"], device,
                         identity["protocol"]["microbatch_size"][identity["task"]], outer_anchors)
    write_csv(job / "evaluation/predictions.csv", rows)
    report = {**identity, "status": "complete", "evaluation_partition": "outer_test",
              "scores": result, "initialization": payload["initialization"],
              "final_checkpoint_sha256": sha256_file(job / "refit/final.pt"),
              "predictions_sha256": sha256_file(job / "evaluation/predictions.csv")}
    write_json(job / "evaluation/report.json", report)
    return report


def verify_job(job, root, identity):
    report = read_json(job / "report.json")
    validate_metadata(report, {k: v for k, v in identity.items() if k != "environment"})
    if report["status"] != "complete" or identity["smoke"]:
        raise RuntimeError("Incomplete full fold")
    for name, sha in report["artifacts"].items():
        if sha256_file(job / name) != sha:
            raise RuntimeError(f"Fold artifact changed: {name}")
    final = final_payload(job / "refit/final.pt", root)
    validate_metadata(final["identity"], {k: v for k, v in identity.items() if k != "environment"})
    if final["role"] != "source_refit" or not final["refit_reset_verified"]:
        raise RuntimeError("Missing source refit")
    rows = read_csv(job / "evaluation/predictions.csv")
    records = [{"sample_id": r["sample_id"], "subject_id": r["subject_id"], "session": r["session"],
                "window_indices": [int(i) for i in str(r["window_indices"]).split(",")]} for r in rows]
    if digest_json(records) != identity["test_samples_sha256"]:
        raise RuntimeError("Outer scored samples changed")
    if set(r["subject_id"] for r in rows) != set(identity["split"]["test_subjects"]):
        raise RuntimeError("Outer scored subjects changed")
    scores = metrics_from_rows(rows)
    evaluation = read_json(job / "evaluation/report.json")
    if any(scores[k] != evaluation["scores"][k] for k in scores) or evaluation["evaluation_partition"] != "outer_test":
        raise RuntimeError("Outer metrics do not reproduce")
    return report, rows


def compact_intermediates(job, root, *, checkpoint_schema=SCHEMA):
    """Remove only this new job's redundant checkpoints after full verified completion."""
    job.resolve().relative_to(root.resolve())
    proof = job / "report.json"
    if not proof.is_file():
        proof = job / "smoke_report.json"
    if not proof.is_file():
        raise ValueError("Cannot compact an incomplete fold")
    completion = read_json(proof)
    if completion.get("checkpoint_schema") != checkpoint_schema or completion.get("status") not in ("complete", "smoke_passed"):
        raise ValueError("No verified full-workflow completion marker")
    removed = []
    for phase, names in (("selection", ("best", "last", "boundary")), ("refit", ("boundary",))):
        for name in names:
            for suffix in (".pt", ".sha256.json"):
                path = job / phase / (name + suffix)
                path.resolve().relative_to(job.resolve())
                if path.exists():
                    path.unlink()
                    removed.append(str(path.relative_to(job)))
    if removed or not (job / "retention.json").exists():
        write_json(job / "retention.json", {"policy": "final_model_and_all_text_audits_retained",
                                            "removed_redundant_weights": removed})


def run_fold(job, root, identity, bank, partitions, device, *, checkpoint_path=None,
             resume=False, encoder_factory=TemporalEncoder):
    if resume and identity["smoke"] and (job / "smoke_report.json").is_file():
        report = read_json(job / "smoke_report.json")
        validate_metadata(report, {k: v for k, v in identity.items() if k != "environment"})
        payload = final_payload(job / "refit/final.pt", root)
        validate_metadata(payload["identity"], {k: v for k, v in identity.items() if k != "environment"})
        if report["refit"]["final_checkpoint_sha256"] != sha256_file(job / "refit/final.pt"):
            raise RuntimeError("Smoke final checkpoint changed")
        compact_intermediates(job, root)
        return report
    if resume and (job / "report.json").is_file():
        report, _ = verify_job(job, root, identity)
        compact_intermediates(job, root)
        return report
    if not resume:
        assert_empty(job)
    train, val, source, test, anchors, outer_anchors = partitions
    selection = train_pilot(job / "selection", identity, bank, train, val, anchors, device,
        checkpoint_path=checkpoint_path, resume=resume, encoder_factory=encoder_factory, checkpoint_schema=SCHEMA)
    refit = run_refit(job / "refit", root, identity, selection, bank, source, anchors, device,
                     checkpoint_path=checkpoint_path, resume=resume, encoder_factory=encoder_factory)
    if identity["smoke"]:
        report = {**identity, "status": "smoke_passed", "selection": selection, "refit": refit}
        write_json(job / "smoke_report.json", report)
        compact_intermediates(job, root)
        return report
    evaluation = evaluate_outer(job, root, identity, bank, test, outer_anchors, device, encoder_factory)
    report = {**identity, "status": "complete", "initialization": selection["initialization"],
              "selection_best_step": selection["best_global_step"], "selection_best_epoch": selection["best_epoch"],
              "selection_validation": selection["best_validation"], "refit_steps": refit["refit_steps"],
              "refit_reset_verified": True, "scores": evaluation["scores"],
              "selection_order_sha256": selection["first_epoch_order_sha256"],
              "refit_order_sha256": refit["first_epoch_order_sha256"],
              "artifacts": {name: sha256_file(job / name) for name in (
                  "selection/report.json", "selection/history.json", "selection/initialization.json",
                  "refit/final.pt", "refit/report.json", "refit/history.json",
                  "evaluation/report.json", "evaluation/predictions.csv")}}
    write_json(job / "report.json", report)
    verify_job(job, root, identity)
    compact_intermediates(job, root)
    return report


def summarize_condition(root, condition, identities):
    expected = set(itertools.product(TASKS, VARIANTS, DATASETS, range(1, 6)))
    actual = {(i["task"], i["variant"], i["dataset"], int(i["fold_id"].split("_")[1])) for i in identities}
    if actual != expected or len(identities) != 120 or any(
        i["condition"] != condition or i["smoke"] or i["checkpoint_schema"] != SCHEMA for i in identities
    ):
        raise ValueError("Condition summary requires all 120 non-smoke fivefold jobs")
    grouped, pairings, fold_table = {}, {}, []
    for identity in identities:
        t, v, d, f = (identity[k] for k in ("task", "variant", "dataset", "fold_id"))
        job = root / condition / t / v / d / f
        report, rows = verify_job(job, root, identity)
        init = report["initialization"]
        checks = {("samples", t, d, f): _row_identity(rows),
                  ("shuffle", t, d, f): [report["selection_order_sha256"], report["refit_order_sha256"]],
                  ("head",): init["head_sha256"], ("shapes",): init["parameter_shapes"],
                  ("encoder", VARIANTS[v]["pretrained"]): init["encoder_sha256"],
                  ("protocol",): identity["protocol"], ("binding",): identity["binding"]}
        for key, value in checks.items():
            if pairings.setdefault(key, value) != value:
                raise RuntimeError(f"Fivefold pairing mismatch: {key}")
        metrics = report["scores"]["metrics"]
        grouped.setdefault((t, v, d), []).append((f, rows, metrics))
        fold_table.append({"condition": condition, "task": t, "variant": v, "dataset": d, "fold": f,
            **{k: metrics[k] for k in ("accuracy", "balanced_accuracy", "AUROC")},
            "test_BCE": report["scores"]["loss"], "selection_best_step": report["selection_best_step"],
            "selection_best_epoch": report["selection_best_epoch"], "refit_steps": report["refit_steps"]})
    dataset_table, pooled_hashes = [], {}
    for (t, v, d), groups in grouped.items():
        rows = [r for _, data, _ in sorted(groups) for r in data]
        if len({r["sample_id"] for r in rows}) != len(rows):
            raise RuntimeError("Outer-test sample appears in more than one fold")
        subject_sets = [set(r["subject_id"] for r in data) for _, data, _ in groups]
        if sum(map(len, subject_sets)) != len(set().union(*subject_sets)):
            raise RuntimeError("Outer-test subject appears in more than one fold")
        pooled = metrics_from_rows(rows)["metrics"]
        pooled_hashes[f"{t}/{v}/{d}"] = digest_json(_row_identity(rows))
        entry = {"condition": condition, "task": t, "variant": v, "dataset": d, "n_samples": len(rows)}
        for metric in ("accuracy", "balanced_accuracy", "AUROC"):
            values = [m[metric] for _, _, m in groups]
            entry[metric] = pooled[metric]
            entry[metric + "_fold_mean"] = None if any(x is None for x in values) else float(np.mean(values))
            entry[metric + "_fold_std"] = None if any(x is None for x in values) else float(np.std(values, ddof=1))
        dataset_table.append(entry)
    macros = []
    for t, v in itertools.product(TASKS, VARIANTS):
        entries = [r for r in dataset_table if r["task"] == t and r["variant"] == v]
        macros.append({"task": t, "variant": v, "condition": condition,
                       **{m: float(np.mean([r[m] for r in entries])) for m in ("accuracy", "balanced_accuracy", "AUROC")}})
    result = {"status": "complete", "condition": condition, "jobs": 120, "full_fivefold_completed": True,
              "protocol": identities[0]["protocol"], "binding": identities[0]["binding"],
              "datasets": dataset_table, "dataset_macros": macros, "pooled_sample_hashes": pooled_hashes,
              "initialization_pairing": {"head": pairings["head",], "shapes": pairings["shapes",],
                  "random_encoder": pairings["encoder", False], "pretrained_encoder": pairings["encoder", True]},
              "limits": identities[0]["protocol"]["limits"]}
    destination = root / "summary" / condition
    write_csv(destination / "fold_metrics.csv", fold_table)
    write_csv(destination / "dataset_metrics.csv", dataset_table)
    write_csv(destination / "dataset_macros.csv", macros)
    write_json(destination / "aggregate_report.json", result)
    return result


def compare_conditions(root):
    reports = {c: read_json(root / "summary" / c / "aggregate_report.json") for c in ("C0", "C1")}
    for c, r in reports.items():
        validate_metadata(r, {"status": "complete", "condition": c, "jobs": 120, "full_fivefold_completed": True})
    for key in ("protocol", "binding", "pooled_sample_hashes", "initialization_pairing"):
        if reports["C0"][key] != reports["C1"][key]:
            raise RuntimeError(f"C0/C1 comparison mismatch: {key}")
    comparisons = {}
    for t in TASKS:
        comparisons[t] = {}
        for d in (*DATASETS, "dataset_macro"):
            per_metric = {}
            for metric in ("accuracy", "balanced_accuracy", "AUROC"):
                values = {}
                for c, r in reports.items():
                    entries = r["dataset_macros"] if d == "dataset_macro" else [e for e in r["datasets"] if e["dataset"] == d]
                    for e in entries:
                        if e["task"] == t:
                            values[c, e["variant"]] = e[metric]
                per_metric[metric] = contrasts(values)
            comparisons[t][d] = per_metric
    result = {"status": "complete", "jobs": 240, "full_fivefold_completed": True,
              "protocol": reports["C0"]["protocol"], "comparisons": comparisons,
              "condition_summary_sha256": {c: sha256_file(root / "summary" / c / "aggregate_report.json") for c in reports},
              "limits": reports["C0"]["limits"]}
    write_json(root / "summary/paired_comparison.json", result)
    return result
