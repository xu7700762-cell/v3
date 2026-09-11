"""Frozen token pilot with verified caches and complete-epoch recovery."""
import copy
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..evaluation.io import read_json, write_json, write_csv, sha256_file
from ..evaluation.anchor_pilot import load_verified, metrics_from_rows
from ..model.token_probe import TokenReadout, FrozenTokenProbe
from ..model.kan import FractionalDoGPolynomialKANLayer
from ..model.linear_probe import build_probe, tensor_state_hash
from ..ssl_protocol import digest_json, assert_empty
from .anchor_pilot import save_checkpoint, rng_state, restore_rng, should_validate
from .probe_data import epoch_order
from .ssl_ablation import seed_runtime
from .token_data import sample_hash

SCHEMA = "femba_reference_free_token_pilot_v1"


def require_space(root, additional=0):
    root = Path(root)
    existing = root
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < additional + 5 * 1024**3:
        raise OSError("D drive must retain 5 GiB after estimated new artifacts; no files were deleted")


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def prepare_encoder(config, root, device):
    bundle = read_json(Path(config["protocol_root"]) / "manifest.json")
    official = bundle["pretrained_femba"]["sha256"]
    model, initial = build_probe("A3", 2001, device, checkpoint_path=config["paths"]["pretrain_checkpoint"],
                                 expected_sha256=official)
    encoder = model.encoder.eval()
    path = Path(root) / "shared/encoder.pt"
    expected = {"official_sha256": official, "encoder_sha256": initial["encoder_sha256"],
                "load_info": initial["pretrain_load_info"]}
    if path.exists():
        payload = load_verified(path)
        if payload["provenance"] != expected or tensor_state_hash(payload["state"]) != initial["encoder_sha256"]:
            raise RuntimeError("Shared encoder provenance changed")
    else:
        require_space(root, 150 * 1024**2)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(path, {"state": cpu_state(encoder), "provenance": expected})
    return encoder, {**expected, "shared_file_sha256": sha256_file(path)}


@torch.no_grad()
def encode_batch(encoder, values, device):
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        result = encoder.forward_tokens(torch.as_tensor(values, dtype=torch.float32, device=device)).float()
    if result.shape[1:] != (80, 525) or not torch.isfinite(result).all():
        raise RuntimeError("Invalid cached encoder tokens")
    return result.cpu().numpy()


def token_cache(root, windows, data_identity, encoder, encoder_identity, device):
    binding = {"data": data_identity, "encoder": encoder_identity,
               "preprocess_code": sha256_file(Path(__file__).with_name("token_data.py")),
               "precision": "encoder_bfloat16_tokens_float32", "encoding_batch": 4}
    key = digest_json(binding)
    folder = Path(root) / "cache" / data_identity["dataset"] / key
    path, manifest = folder / "tokens.npy", folder / "manifest.json"
    before = tensor_state_hash(encoder.state_dict())
    if manifest.exists():
        report = read_json(manifest)
        if report["binding"] != binding or sha256_file(path) != report["tokens_sha256"]:
            raise RuntimeError("Token cache integrity mismatch")
    else:
        require_space(root, windows.shape[0] * 80 * 525 * 4 + 800 * 1024**2)
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / "tokens.partial.npy"
        out = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(windows), 80, 525))
        began = time.perf_counter()
        for start in range(0, len(windows), 4):
            out[start:start + 4] = encode_batch(encoder, windows[start:start + 4], device)
        out.flush()
        del out
        os.replace(temporary, path)
        report = {"binding": binding, "tokens_sha256": sha256_file(path), "encoding_seconds": time.perf_counter() - began,
                  "encoded_windows": len(windows), "shape": [len(windows), 80, 525]}
        write_json(manifest, report)
    tokens = np.load(path, mmap_mode="r")
    if list(tokens.shape) != report["shape"] or tokens.dtype != np.float32:
        raise RuntimeError("Wrong cache shape or dtype")
    # Independent re-encoding uses the same fixed batch geometry as cache creation.
    for start in sorted({0, (len(windows) // 8) * 4, ((len(windows) - 1) // 4) * 4}):
        fresh = encode_batch(encoder, windows[start:start + 4], device)
        if not np.array_equal(fresh, tokens[start:start + 4]):
            raise RuntimeError("Cached and online encoding differ")
    if before != tensor_state_hash(encoder.state_dict()) or any(p.grad is not None for p in encoder.parameters()):
        raise RuntimeError("Frozen encoder changed during encoding")
    return tokens, {"key": key, "manifest_sha256": sha256_file(manifest),
                    "tokens_sha256": report["tokens_sha256"], "online_verified": True,
                    "encoded_windows": report["encoded_windows"], "encoding_seconds": report["encoding_seconds"]}


def token_inputs(tokens, examples, task, device):
    indices = np.asarray([e.indices for e in examples], dtype=np.int64)
    value = np.asarray(tokens[indices])
    if task == "state":
        value = value[:, 0]
    return torch.as_tensor(value, dtype=torch.float32, device=device)


@torch.no_grad()
def score_head(head, tokens, examples, device, micro):
    head.eval()
    rows = []
    for start in range(0, len(examples), micro):
        batch = examples[start:start + micro]
        logits = head(token_inputs(tokens, batch, head.task, device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite validation logits")
        for e, logit, probability in zip(batch, logits.cpu().tolist(), logits.sigmoid().cpu().tolist()):
            rows.append({"sample_id": e.sample_id, "subject_id": e.subject_id, "session": e.session,
                         "window_indices": ",".join(map(str, e.indices)), "logit": logit, "score": probability,
                         "threshold": 0.5, "y_pred": int(probability >= 0.5)})
    # Attach labels after inference; no label or subject identity is passed to the head.
    labels = {e.sample_id: e.label for e in examples}
    rows = [{**r, "y_true": labels[r["sample_id"]], "correct": int(r["y_pred"] == labels[r["sample_id"]])} for r in rows]
    result = metrics_from_rows(rows)
    result["metrics"]["n_subjects"] = len({e.subject_id for e in examples})
    return rows, result


def optimizer_step(head, optimizer, tokens, batch, device, protocol, weight):
    head.train()
    optimizer.zero_grad(set_to_none=True)
    micro = protocol["microbatch_size"][head.task]
    total = 0.0
    for start in range(0, len(batch), micro):
        group = batch[start:start + micro]
        logits = head(token_inputs(tokens, group, head.task, device))
        labels = torch.tensor([e.label for e in group], dtype=torch.float32, device=device)
        loss = F.binary_cross_entropy_with_logits(logits, labels,
                    pos_weight=torch.tensor(weight, device=device), reduction="sum") / len(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        total += loss.item()
    norms = {}
    for name, p in head.named_parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                raise FloatingPointError(f"Non-finite gradient: {name}")
            norms[name] = p.grad.norm().item()
    norm = torch.nn.utils.clip_grad_norm_([p for p in head.parameters() if p.requires_grad],
                                         protocol["grad_clip"], error_if_nonfinite=True)
    optimizer.step()
    return {"loss": total, "gradient_norm": float(norm), "parameter_gradient_norms": norms,
            "targets": len(batch), "cached_window_reads": sum(len(example.indices) for example in batch)}


@torch.no_grad()
def basis_diagnostics(head, tokens, examples, device, micro):
    if not hasattr(head, "mapping") or not isinstance(head.mapping[1], FractionalDoGPolynomialKANLayer):
        return {}
    layer = head.mapping[1]
    state = cpu_state(head)
    values = token_inputs(tokens, examples[:1], head.task, device).reshape(-1, 525)[:128]
    q = head.mapping[0](values)
    first, second, fractional, dog = layer.basis_features(q)
    contribution = .25 * layer.dog_mix_logits.tanh() * dog
    diagnostics = {"fractional_order": float(1 + .25 * layer.fractional_order_logit.tanh()),
                   "dog_mix_abs_mean": float((.25 * layer.dog_mix_logits.tanh()).abs().mean()),
                   "dog_mix_abs_max": float((.25 * layer.dog_mix_logits.tanh()).abs().max()),
                   "dog_term_rms": float(contribution.square().mean().sqrt()),
                   "first_basis_rms": float(first.square().mean().sqrt()),
                   "second_basis_rms": float(second.square().mean().sqrt()),
                   "diagnostic_tokens": len(q), "scope": "first_validation_sample_up_to_128_tokens"}
    for mode in ("no_dog", "order1", "order1_no_dog"):
        head.load_state_dict(state)
        if "no_dog" in mode:
            layer.dog_mix_logits.zero_()
        if "order1" in mode:
            layer.fractional_order_logit.zero_()
        _, diagnostics["counterfactual_" + mode] = score_head(head, tokens, examples, device, micro)
    head.load_state_dict(state)
    return diagnostics


def verify_artifacts(folder, identity=None):
    folder = Path(folder)
    report = read_json(folder / "report.json")
    if identity is not None and report["identity"] != identity:
        raise RuntimeError("Completed pilot identity changed")
    required = {"best.pt", "last.pt", "boundary.pt", "history.json", "initialization.json", "predictions.csv", "diagnostics.json"}
    if not required <= set(report["artifacts"]):
        raise RuntimeError("Missing required pilot artifacts")
    for name, sha in report["artifacts"].items():
        if sha256_file(folder / name) != sha:
            raise RuntimeError(f"Pilot artifact hash mismatch: {name}")
    if report["status"] not in ("complete", "smoke_passed") or report["reload_max_abs_error"] != 0:
        raise RuntimeError("Incomplete or invalid pilot")
    return report


def _new_head(identity, device, head_factory=None):
    if head_factory is None:
        head = TokenReadout(identity["config"], identity["task"], identity["protocol"]["seed"])
    else:
        head = head_factory(identity["config"], identity["protocol"]["seed"])
    return head.to(device)


def restored_head(path, identity, device, head_factory=None):
    payload = load_verified(path)
    if payload["identity"] != identity:
        raise RuntimeError("Checkpoint identity mismatch")
    head = _new_head(identity, device, head_factory)
    if head.initialization() != payload["initialization"]:
        raise RuntimeError("Head initialization changed")
    head.load_state_dict(payload["state"], strict=True)
    if tensor_state_hash(head.state_dict()) != payload["state_sha256"]:
        raise RuntimeError("Head tensor integrity mismatch")
    return head.eval(), payload


def train_job(folder, identity, tokens, train, val, device, *, resume=False, step_callback=None,
              head_factory=None, diagnostics_fn=None, optimizer_step_fn=None, score_fn=None):
    folder = Path(folder)
    if identity["train_sha256"] != sample_hash(train) or identity["val_sha256"] != sample_hash(val):
        raise RuntimeError("Training/validation samples changed")
    if {e.subject_id for e in train} & {e.subject_id for e in val}:
        raise RuntimeError("Training/validation identities overlap")
    if resume and (folder / "report.json").exists():
        return verify_artifacts(folder, identity)
    if not resume:
        assert_empty(folder)
    require_space(folder, 30 * 1024**2)
    folder.mkdir(parents=True, exist_ok=True)
    protocol = identity["protocol"]
    seed_runtime(protocol["seed"])
    head = _new_head(identity, device, head_factory)
    initial = head.initialization()
    optimizer = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],
                                  lr=protocol["head_lr"], weight_decay=protocol["weight_decay"])
    batch_size, micro = protocol["batch_size"][head.task], protocol["microbatch_size"][head.task]
    n = math.ceil(len(train) / batch_size)
    positive = sum(e.label for e in train)
    if not 0 < positive < len(train):
        raise ValueError("Training partition must contain both classes")
    weight = (len(train) - positive) / positive
    step, epoch, best, history, events = 0, 0, None, [], []
    totals = {"optimizer_seconds": 0.0, "validation_seconds": 0.0, "cached_training_window_reads": 0,
              "cached_validation_window_reads": 0, "seen_targets": 0}
    max_norms = {}
    begin = time.perf_counter()
    score = score_fn or score_head
    take_step = optimizer_step_fn or optimizer_step

    def payload(role, validation=None):
        state = cpu_state(head)
        return {"identity": identity, "initialization": initial, "role": role,
                "step": step, "epoch": epoch, "state": state, "state_sha256": tensor_state_hash(state),
                "validation": validation}

    def boundary():
        saved = payload("boundary")
        saved.update(optimizer=optimizer.state_dict(), rng=rng_state(), best=best, history=history,
                     events=events, totals=totals, max_norms=max_norms)
        save_checkpoint(folder / "boundary.pt", saved)

    boundary_path = folder / "boundary.pt"
    if resume and (boundary_path.exists() or boundary_path.with_suffix(".previous.pt").exists()):
        saved = load_verified(boundary_path, recover=True)
        if saved["identity"] != identity or saved["initialization"] != initial or saved["step"] % n or saved["role"] != "boundary":
            raise RuntimeError("Invalid complete-epoch resume boundary")
        if tensor_state_hash(saved["state"]) != saved["state_sha256"]:
            raise RuntimeError("Boundary head hash mismatch")
        head.load_state_dict(saved["state"])
        optimizer.load_state_dict(saved["optimizer"])
        step, epoch, best = saved["step"], saved["epoch"], saved["best"]
        history, events, totals, max_norms = saved["history"], saved["events"], saved["totals"], saved["max_norms"]
        events.append({"restored_epoch": epoch, "restored_step": step, "policy": "rollback_history_and_best"})
        if best is not None:
            if tensor_state_hash(best["state"]) != best["state_sha256"]:
                raise RuntimeError("Boundary best hash mismatch")
            save_checkpoint(folder / "best.pt", best)
        elif (folder / "best.pt").exists():
            (folder / "best.pt").rename(folder / f"best.abandoned-{time.time_ns()}.pt")
        write_json(folder / "history.json", {"checks": history, "resume_events": events})
        restore_rng(saved["rng"])
    else:
        if resume and any(folder.iterdir()):
            raise RuntimeError("Nonempty pilot directory has no valid resume boundary")
        write_json(folder / "initialization.json", initial)
        boundary()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    limit = 2 if identity["smoke"] else n * protocol["max_epochs"]
    stopped = best is not None and not identity["smoke"] and step - best["step"] >= n * protocol["patience_epochs"]
    while step < limit and not stopped:
        epoch += 1
        order = epoch_order(len(train), protocol["seed"], epoch)
        for start in range(0, len(order), batch_size):
            batch = [train[int(i)] for i in order[start:start + batch_size]]
            if device.type == "cuda":
                torch.cuda.synchronize()
            began = time.perf_counter()
            training = take_step(head, optimizer, tokens, batch, device, protocol, weight)
            if device.type == "cuda":
                torch.cuda.synchronize()
            totals["optimizer_seconds"] += time.perf_counter() - began
            totals["cached_training_window_reads"] += training["cached_window_reads"]
            totals["seen_targets"] += len(batch)
            for name, value in training["parameter_gradient_norms"].items():
                max_norms[name] = max(max_norms.get(name, 0), value)
            step += 1
            row = {"step": step, "epoch": epoch, "training": training}
            if should_validate(step, n) or step == limit:
                began = time.perf_counter()
                _, validation = score(head, tokens, val, device, micro)
                totals["validation_seconds"] += time.perf_counter() - began
                totals["cached_validation_window_reads"] += sum(len(example.indices) for example in val)
                row["validation"] = validation
                if best is None or validation["loss"] < best["validation"]["loss"]:
                    best = payload("best", validation)
                    save_checkpoint(folder / "best.pt", best)
                stopped = not identity["smoke"] and step - best["step"] >= protocol["patience_epochs"] * n
                print(f"{identity['dataset']}/{head.task}/{head.config} epoch={epoch} step={step} "
                      f"val_BCE={validation['loss']:.6f} best={best['step']}", flush=True)
            history.append(row)
            if "validation" in row or step % n == 0:
                write_json(folder / "history.json", {"checks": history, "resume_events": events})
            if step % n == 0:
                boundary()
            if step_callback:
                step_callback(step)
            if step >= limit or stopped:
                break
    if best is None:
        raise RuntimeError("No best checkpoint was selected")
    last_rows, last_validation = score(head, tokens, val, device, micro)
    save_checkpoint(folder / "last.pt", payload("last", last_validation))
    updated = tensor_state_hash(head.state_dict()) != initial["head_sha256"]
    if not updated or not any(v > 0 for v in max_norms.values()):
        raise RuntimeError("Head did not update")
    head.load_state_dict(best["state"])
    rows, best_validation = score(head, tokens, val, device, micro)
    if best_validation != best["validation"]:
        raise RuntimeError("Best model no longer reproduces selection")
    reload, _ = restored_head(folder / "best.pt", identity, device, head_factory)
    again, reloaded_metrics = score(reload, tokens, val, device, micro)
    if rows != again or best_validation != reloaded_metrics:
        raise RuntimeError("Checkpoint reload changed predictions")
    diagnostics = (diagnostics_fn or basis_diagnostics)(reload, tokens, val, device, micro)
    diagnostics["maximum_gradient_norms"] = max_norms
    write_json(folder / "diagnostics.json", diagnostics)
    write_csv(folder / "predictions.csv", rows)
    write_csv(folder / "last_predictions.csv", last_rows)
    write_json(folder / "history.json", {"checks": history, "resume_events": events})
    first_order = [train[int(i)].sample_id for i in epoch_order(len(train), protocol["seed"], 1)]
    report = {"identity": identity, "status": "smoke_passed" if identity["smoke"] else "complete",
              "initialization": initial, "global_step": step, "best_step": best["step"], "best_epoch": best["epoch"],
              "best_validation": best_validation, "last_validation": last_validation, "steps_per_epoch": n,
              "train_pos_weight": weight, "head_updated": updated, "reload_max_abs_error": 0,
              "first_epoch_order_sha256": digest_json(first_order), "resume_events": events,
              "encoder_actual_training_forwards": 0, "frozen_token_cache": True,
              "invocation_seconds": time.perf_counter() - begin,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
              **totals, "artifacts": {name: sha256_file(folder / name) for name in (
                  "best.pt", "last.pt", "boundary.pt", "history.json", "initialization.json",
                  "predictions.csv", "last_predictions.csv", "diagnostics.json")}}
    write_json(folder / "report.json", report)
    return verify_artifacts(folder, identity)


def online_smoke(head, encoder, windows, tokens, device):
    """Exercise parent train(), actual GPU encoder and cached/online head predictions."""
    model = FrozenTokenProbe(encoder, head).to(device).train()
    before = tensor_state_hash(encoder.state_dict())
    # Fixed batch geometry mirrors token_cache. Severity checks every independently cached window.
    count = 4 if head.task == "state" else int(
        getattr(head, "smoke_window_count", getattr(head, "window_count", 11))
    )
    with torch.no_grad():
        encoded = np.concatenate([encode_batch(encoder, windows[i:i + 4], device) for i in range(0, count, 4)])
        a = torch.tensor(encoded if head.task == "state" else encoded[:count][None], device=device)
        b = torch.tensor(np.array(tokens[:count] if head.task == "state" else tokens[:count][None]), device=device)
        error = float((head(a) - head(b)).abs().max())
    if error != 0 or encoder.training or any(p.grad is not None for p in encoder.parameters()) or before != tensor_state_hash(encoder.state_dict()):
        raise RuntimeError("Frozen encoder online smoke failed")
    return {"online_cache_max_abs_error": error, "encoder_unchanged": True, "encoder_training": False}
