from pathlib import Path
import math
import os
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..evaluation.io import write_json, read_json, sha256_file
from ..evaluation.anchor_pilot import score, load_verified, restore_checkpoint
from ..anchor_protocol import SCHEMA
from ..model.anchor_probe import build_anchor_probe
from ..model.linear_probe import tensor_state_hash
from ..ssl_protocol import assert_empty, digest_json, validate_metadata
from .anchor_data import batch_inputs
from .probe_data import epoch_order, class_weight, sample_digest
from .ssl_ablation import optimizer_for, seed_runtime, snapshot, update_audit


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    manifest = path.with_suffix(".sha256.json")
    temporary_manifest = path.with_suffix(".tmp.sha256.json")
    previous = path.with_suffix(".previous.pt")
    previous_manifest = path.with_suffix(".previous.sha256.json")
    torch.save(payload, temporary)
    write_json(temporary_manifest, {"sha256": sha256_file(temporary)})
    # Retain the previous committed pair until both new files have committed.
    if path.exists():
        os.replace(path, previous)
    if manifest.exists():
        os.replace(manifest, previous_manifest)
    os.replace(temporary, path)
    os.replace(temporary_manifest, manifest)
    previous.unlink(missing_ok=True)
    previous_manifest.unlink(missing_ok=True)


def rng_state():
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def should_validate(step, steps_per_epoch):
    return step == 1 or step % max(1, math.ceil(steps_per_epoch / 4)) == 0 or step % steps_per_epoch == 0


def is_better(loss, best):
    return best is None or loss < best["validation"]["loss"]


def optimizer_step(model, optimizer, bank, batch, task, device, protocol, anchors, weight):
    model.train()
    if model.encoder.training != model.encoder_trainable:
        raise AssertionError("Encoder train/eval contract failed")
    optimizer.zero_grad(set_to_none=True)
    loss_total, encoded = 0.0, 0
    micro = protocol["microbatch_size"][task]
    for start in range(0, len(batch), micro):
        examples = batch[start:start + micro]
        inputs, count = batch_inputs(bank, examples, task, device, model.condition, anchors)
        encoded += count
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(*inputs).float()
        labels = torch.tensor([e.label for e in examples], dtype=torch.float32, device=device)
        loss = F.binary_cross_entropy_with_logits(logits, labels,
            pos_weight=torch.tensor(weight, dtype=torch.float32, device=device), reduction="sum") / len(batch)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite pilot loss")
        loss.backward()
        loss_total += float(loss.detach())
    received, head_received, norms = [], [], {}
    for prefix, module in (("encoder", model.encoder), ("head", model.head)):
        squared = torch.zeros((), device=device)
        for name, p in module.named_parameters():
            if p.grad is None:
                continue
            if prefix == "encoder" and not model.encoder_trainable:
                raise AssertionError("Frozen encoder received gradient")
            if not bool(torch.isfinite(p.grad).all()):
                raise FloatingPointError("Non-finite gradient")
            squared += p.grad.float().square().sum()
            if prefix == "encoder" and bool(torch.any(p.grad != 0)):
                received.append(name)
            if prefix == "head" and bool(torch.any(p.grad != 0)):
                head_received.append(name)
        norms[prefix] = float(squared.sqrt())
    torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["grad_clip"], error_if_nonfinite=True)
    optimizer.step()
    return {"loss": loss_total, "examples": len(batch), "encoded_windows": encoded,
            "gradient_norms": norms, "encoder_gradient_names": received,
            "head_gradient_names": head_received}


def train_pilot(output, identity, bank, train, val, anchors, device, *, checkpoint_path=None,
                resume=False, encoder_factory=None, step_callback=None, checkpoint_schema=SCHEMA,
                model_factory=None):
    invocation_started = time.perf_counter()
    if identity["checkpoint_schema"] != checkpoint_schema:
        raise ValueError("Selection checkpoint schema mismatch")
    output = Path(output)
    for name, examples, hash_key in (("source_train_subjects", train, "train_samples_sha256"),
                                     ("source_val_subjects", val, "val_samples_sha256")):
        if set(e.subject_id for e in examples) != set(identity["split"][name]) or sample_digest(examples) != identity[hash_key]:
            raise ValueError("Pilot partition/sample mismatch")
    if set(identity["split"]["source_train_subjects"]) & set(identity["split"]["source_val_subjects"]):
        raise ValueError("Pilot train/val overlap")
    if digest_json(anchors) != identity["anchors_sha256"]:
        raise ValueError("Anchor identity mismatch")
    expected = {k: v for k, v in identity.items() if k != "environment"}
    if resume and (output / "report.json").is_file():
        report = read_json(output / "report.json")
        validate_metadata(report, expected)
        for name, sha in report["artifacts"].items():
            if sha256_file(output / name) != sha:
                raise RuntimeError("Completed training artifact changed")
        return report
    if not resume:
        assert_empty(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol, task = identity["protocol"], identity["task"]
    seed_runtime(identity["training_seed"])
    kwargs = {} if encoder_factory is None else {"encoder_factory": encoder_factory}
    if model_factory is not None:
        kwargs["model_factory"] = model_factory
    model, initial = build_anchor_probe(identity["variant"], identity["training_seed"], device,
        condition=identity["condition"], checkpoint_path=checkpoint_path,
        expected_sha256=identity["official_pretrain_sha256"], **kwargs)
    before = snapshot(model)
    optimizer = optimizer_for(model, protocol)
    batch_size = protocol["batch_size"][task]
    n = math.ceil(len(train) / batch_size)
    weight = class_weight(train)
    history, best, global_step, completed_epoch = [], None, 0, 0
    received, totals = set(), {"training_seconds": 0.0, "training_encoded_windows": 0,
                              "validation_encoded_windows": 0, "seen_targets": 0}
    max_norms, resume_events = {"encoder": 0.0, "head": 0.0}, []
    first_order = digest_json([train[int(i)].sample_id for i in epoch_order(len(train), identity["training_seed"], 1)])

    def payload(role, epoch, validation=None):
        state = snapshot(model)
        return {"identity": identity, "initialization": initial, "role": role,
                "model_state_dict": state, "model_sha256": tensor_state_hash(state),
                "global_step": global_step, "epoch": epoch, "validation": validation}

    def boundary(epoch):
        value = payload("boundary", epoch)
        value.update({"optimizer": optimizer.state_dict(), "rng": rng_state(), "best": best,
                      "history": history, "received": sorted(received), "totals": totals,
                      "max_norms": max_norms, "branch_gradients": model.branch_gradients,
                      "resume_events": resume_events})
        save_checkpoint(output / "boundary.pt", value)

    boundary_path = output / "boundary.pt"
    if resume and (boundary_path.is_file() or boundary_path.with_suffix(".previous.pt").is_file()):
        saved = load_verified(boundary_path, recover=True)
        validate_metadata(saved["identity"], expected)
        if saved["initialization"] != initial or saved["role"] != "boundary" or saved["global_step"] % n:
            raise RuntimeError("Invalid epoch boundary")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        if tensor_state_hash(model.state_dict()) != saved["model_sha256"]:
            raise RuntimeError("Boundary model hash mismatch")
        optimizer.load_state_dict(saved["optimizer"])
        history, best = saved["history"], saved["best"]
        global_step, completed_epoch = saved["global_step"], saved["epoch"]
        received, totals = set(saved["received"]), saved["totals"]
        max_norms, model.branch_gradients = saved["max_norms"], saved["branch_gradients"]
        resume_events = saved["resume_events"] + [{"restored_epoch": completed_epoch, "restored_step": global_step,
            "policy": "rollback_history_and_best_to_complete_epoch_boundary"}]
        if best is not None:
            if tensor_state_hash(best["model_state_dict"]) != best["model_sha256"]:
                raise RuntimeError("Boundary best model hash mismatch")
            save_checkpoint(output / "best.pt", best)
        elif (output / "best.pt").exists():
            # Preserve an abandoned partial-epoch best for audit; never select it.
            suffix = f".abandoned-{time.time_ns()}"
            (output / "best.pt").rename(output / ("best.pt" + suffix))
            sidecar = output / "best.sha256.json"
            if sidecar.exists():
                sidecar.rename(output / ("best.sha256.json" + suffix))
        write_json(output / "history.json", {"checks": history, "resume_events": resume_events})
        restore_rng(saved["rng"])
        del saved
    else:
        if resume and any(output.iterdir()):
            raise RuntimeError("Nonempty interrupted run has no valid boundary; preserve it and use a new output root")
        write_json(output / "initialization.json", initial)
        boundary(0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    target_steps = 2 if identity["smoke"] else protocol["max_epochs"] * n
    stopped = (not identity["smoke"] and best is not None and
               global_step - best["global_step"] >= protocol["patience_epochs"] * n)
    epoch = completed_epoch
    while global_step < target_steps and not stopped:
        epoch += 1
        order = epoch_order(len(train), identity["training_seed"], epoch)
        for start in range(0, len(order), batch_size):
            batch = [train[int(i)] for i in order[start:start + batch_size]]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            began = time.perf_counter()
            step = optimizer_step(model, optimizer, bank, batch, task, device, protocol, anchors, weight)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            totals["training_seconds"] += time.perf_counter() - began
            totals["training_encoded_windows"] += step["encoded_windows"]
            totals["seen_targets"] += step["examples"]
            global_step += 1
            received.update(step["encoder_gradient_names"])
            for key in max_norms:
                max_norms[key] = max(max_norms[key], step["gradient_norms"][key])
            step_row = {"epoch": epoch, "global_step": global_step, "training": step}
            if should_validate(global_step, n) or global_step == target_steps:
                _, validation = score(model, bank, val, task, device, protocol["microbatch_size"][task], anchors)
                totals["validation_encoded_windows"] += validation["encoded_windows"]
                step_row["validation"] = validation
                if is_better(validation["loss"], best):
                    best = payload("best", epoch, validation)
                    save_checkpoint(output / "best.pt", best)
                print(f"{task}/{identity['condition']}/{identity['variant']}/{identity['dataset']} "
                      f"epoch={epoch} step={global_step} train={step['loss']:.6f} "
                      f"val={validation['loss']:.6f} best_step={best['global_step']}", flush=True)
                if not identity["smoke"] and global_step - best["global_step"] >= protocol["patience_epochs"] * n:
                    stopped = True
            history.append(step_row)
            write_json(output / "history.json", {"checks": history, "resume_events": resume_events})
            if step_callback is not None:
                step_callback(global_step)
            if global_step % n == 0:
                boundary(epoch)
            if stopped or global_step >= target_steps:
                break
    if best is None:
        raise AssertionError("No eligible best checkpoint")
    audit = update_audit(model, before)
    if model.encoder_trainable:
        if received != set(dict(model.encoder.named_parameters())):
            raise AssertionError("Some encoder parameters never received nonzero gradients")
        required = ("target", "anchor") if identity["condition"] == "C1" else ("target",)
        if not all(model.branch_gradients[k] for k in required):
            raise AssertionError("Missing encoder branch gradients")
    elif any(model.branch_gradients.values()) or received:
        raise AssertionError("Frozen encoder branch gradient detected")
    audit.update({"encoder_gradient_names": sorted(received), "gradient_norms": max_norms,
                  "branch_gradients": model.branch_gradients,
                  "encoder_parameter_tensor_count": len(list(model.encoder.parameters()))})
    _, last_validation = score(model, bank, val, task, device, protocol["microbatch_size"][task], anchors)
    save_checkpoint(output / "last.pt", payload("last", epoch, last_validation))
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    model.load_state_dict(best["model_state_dict"], strict=True)
    first_rows, best_validation = score(model, bank, val, task, device, protocol["microbatch_size"][task], anchors)
    if best_validation != best["validation"]:
        raise AssertionError("Best checkpoint state does not reproduce selected validation")
    del optimizer, model, before, best
    reloaded, _ = restore_checkpoint(output / "best.pt", identity, device, expected_schema=checkpoint_schema, **kwargs)
    second_rows, reloaded_validation = score(reloaded, bank, val, task, device, protocol["microbatch_size"][task], anchors)
    error = max(abs(a["logit"] - b["logit"]) for a, b in zip(first_rows, second_rows))
    if error != 0 or reloaded_validation != best_validation:
        raise AssertionError("Checkpoint reload changed validation predictions")
    selected = min((r for r in history if "validation" in r), key=lambda r: (r["validation"]["loss"], r["global_step"]))
    report = {**identity, "status": "smoke_passed" if identity["smoke"] else "complete",
              "initialization": initial, "audit": audit, "global_step": global_step,
              "best_global_step": selected["global_step"], "best_epoch": selected["epoch"],
              "best_validation": best_validation, "last_validation": last_validation,
              "train_pos_weight": weight, "steps_per_epoch": n, "peak_cuda_bytes": peak,
              "first_epoch_order_sha256": first_order, "reload_max_abs_error": error,
              "resume_events": resume_events, **totals,
              "invocation_wall_seconds": time.perf_counter() - invocation_started,
              "artifacts": {f: sha256_file(output / f) for f in ("best.pt", "last.pt", "boundary.pt", "history.json", "initialization.json")}}
    write_json(output / "report.json", report)
    return report
