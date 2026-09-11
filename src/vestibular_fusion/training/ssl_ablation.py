from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..evaluation.io import write_json
from ..evaluation.ssl_ablation import score, predict, restore_checkpoint
from ..model.linear_probe import build_probe, tensor_state_hash
from ..ssl_protocol import assert_empty, digest_json
from .probe_data import class_weight, epoch_order, input_tensor, sample_digest


def seed_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def optimizer_for(model, protocol):
    groups = [{"params": list(model.head.parameters()), "lr": protocol["head_lr"]}]
    if model.encoder_trainable:
        groups.append({"params": list(model.encoder.parameters()), "lr": protocol["encoder_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=protocol["weight_decay"])
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if optimized != expected:
        raise AssertionError("Optimizer scope does not equal trainable parameter scope")
    return optimizer


def snapshot(model):
    return {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}


def update_audit(model, before: dict) -> dict:
    result = {}
    for prefix in ("encoder", "head"):
        squared = 0.0
        changed = []
        for name, value in model.state_dict().items():
            if name.startswith(prefix + "."):
                old, new = before[name], value.detach().cpu()
                if not torch.equal(old, new):
                    changed.append(name)
                squared += float((new.double() - old.double()).square().sum())
        result[prefix + "_delta_l2"] = squared ** 0.5
        result[prefix + "_changed_tensors"] = changed
    if result["head_delta_l2"] <= 0:
        raise AssertionError("Linear head did not update")
    if model.encoder_trainable != (result["encoder_delta_l2"] > 0):
        raise AssertionError(f"Encoder freeze/update audit failed: {result}")
    result["encoder_training_mode"] = model.encoder.training
    result["encoder_trainable_parameters"] = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    return result


def train_epoch(model, optimizer, bank, examples, task, device, protocol, epoch,
                pos_weight: float, *, max_steps: int | None = None) -> dict:
    model.train()
    if model.encoder.training != model.encoder_trainable:
        raise AssertionError("Encoder train/eval contract failed")
    order = epoch_order(len(examples), protocol["training_seed"], epoch)
    batch_size = protocol["batch_size"][task]
    total_loss, seen, steps = 0.0, 0, 0
    max_norm = {"encoder": 0.0, "head": 0.0}
    received = set()
    used_ids = []
    started = time.perf_counter()
    for start in range(0, len(order), batch_size):
        batch = [examples[int(i)] for i in order[start:start + batch_size]]
        windows = input_tensor(bank, batch, task, device)
        labels = torch.tensor([e.label for e in batch], dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(windows).float()
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=torch.tensor(pos_weight, device=device)
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite SSL ablation loss")
        loss.backward()
        for prefix, module in (("encoder", model.encoder), ("head", model.head)):
            squared = 0.0
            for name, parameter in module.named_parameters():
                grad = parameter.grad
                if grad is not None:
                    if not bool(torch.isfinite(grad).all()):
                        raise FloatingPointError(f"Non-finite gradient: {prefix}.{name}")
                    squared += float(grad.detach().float().square().sum())
                    if prefix == "encoder" and bool(torch.any(grad != 0)):
                        received.add(name)
                    if prefix == "encoder" and not model.encoder_trainable:
                        raise AssertionError("Frozen encoder received a gradient")
            max_norm[prefix] = max(max_norm[prefix], squared ** 0.5)
        torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        total_loss += float(loss.detach()) * len(batch)
        seen += len(batch)
        steps += 1
        used_ids.extend(e.sample_id for e in batch)
        if max_steps is not None and steps >= max_steps:
            break
    if not steps or max_norm["head"] <= 0:
        raise AssertionError("Training did not produce head gradients")
    if model.encoder_trainable and max_norm["encoder"] <= 0:
        raise AssertionError("Trainable encoder did not receive gradients")
    return {"epoch": epoch, "loss": total_loss / seen, "steps": steps, "examples": seen,
            "elapsed_seconds": time.perf_counter() - started,
            "max_grad_norm": max_norm, "encoder_nonzero_grad_names": sorted(received),
            "sample_order_sha256": digest_json(used_ids)}


def choose_epoch(history, protocol) -> int:
    eligible = [r for r in history if r["epoch"] >= protocol["min_epochs"]]
    if not eligible:
        raise ValueError("No eligible validation epochs")
    return min(eligible, key=lambda r: (r["validation"]["loss"], r["epoch"]))["epoch"]


def train_fold(output: Path, identity: dict, bank, train_examples, val_examples,
               source_examples, test_examples, device, checkpoint_path=None,
               encoder_factory=None) -> dict:
    for name, examples in (("source_train_subjects", train_examples),
                           ("source_val_subjects", val_examples),
                           ("source_subjects", source_examples),
                           ("test_subjects", test_examples)):
        if {e.subject_id for e in examples} != set(identity["split"][name]):
            raise ValueError(f"Examples do not match the locked {name} partition")
    split = identity["split"]
    train_ids, val_ids, test_ids = (set(split[k]) for k in (
        "source_train_subjects", "source_val_subjects", "test_subjects"))
    if train_ids & val_ids or (train_ids | val_ids) & test_ids or (
        train_ids | val_ids != set(split["source_subjects"])
    ):
        raise ValueError("Training, validation and test identities must be disjoint")
    assert_empty(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol, seed = identity["protocol"], identity["training_seed"]
    task, variant, smoke = identity["task"], identity["variant"], identity["smoke"]
    kwargs = {} if encoder_factory is None else {"encoder_factory": encoder_factory}

    def fresh():
        seed_runtime(seed)
        return build_probe(variant, seed, device, checkpoint_path=checkpoint_path,
                           expected_sha256=identity["official_pretrain_sha256"], **kwargs)

    model, initial = fresh()
    write_json(output / "initialization.json", initial)
    selection_history = []
    train_weight = class_weight(train_examples)
    if not smoke:
        optimizer = optimizer_for(model, protocol)
        stale, best_loss = 0, float("inf")
        for epoch in range(1, protocol["max_epochs"] + 1):
            training = train_epoch(model, optimizer, bank, train_examples, task, device,
                                   protocol, epoch, train_weight)
            _, validation = score(model, bank, val_examples, task, device, protocol["batch_size"][task])
            selection_history.append({"epoch": epoch, "training": training, "validation": validation})
            write_json(output / "selection_history.json", selection_history)
            print(f"{task}/{variant}/{identity['dataset']}/{identity['fold_id']} "
                  f"select epoch={epoch} train={training['loss']:.6f} val={validation['loss']:.6f}", flush=True)
            if epoch >= protocol["min_epochs"]:
                if validation["loss"] < best_loss:
                    best_loss, stale = validation["loss"], 0
                else:
                    stale += 1
                if stale >= protocol["patience"]:
                    break
        best_epoch = choose_epoch(selection_history, protocol)
        del optimizer, model
        model, refit_initial = fresh()
        if refit_initial != initial:
            raise AssertionError("Refit did not restart from the original initialization")
        refit_examples = source_examples
    else:
        best_epoch = 0
        refit_examples = train_examples
    before = snapshot(model)
    optimizer = optimizer_for(model, protocol)
    weight = class_weight(refit_examples)
    refit_history = []
    if smoke:
        # Small severity folds may have fewer than two batches. Continue into a
        # second shuffled epoch instead of duplicating an example inside an epoch.
        total_steps, epoch = 0, 0
        while total_steps < 2:
            epoch += 1
            row = train_epoch(model, optimizer, bank, refit_examples, task, device,
                              protocol, epoch, weight, max_steps=2 - total_steps)
            refit_history.append(row)
            total_steps += row["steps"]
    else:
        for epoch in range(1, best_epoch + 1):
            row = train_epoch(model, optimizer, bank, refit_examples, task, device, protocol, epoch, weight)
            refit_history.append(row)
            write_json(output / "refit_history.json", refit_history)
            print(f"{task}/{variant}/{identity['dataset']}/{identity['fold_id']} "
                  f"refit epoch={epoch}/{best_epoch} loss={row['loss']:.6f}", flush=True)
    audit = update_audit(model, before)
    received = set().union(*(set(r["encoder_nonzero_grad_names"]) for r in refit_history))
    audit["encoder_nonzero_grad_names"] = sorted(received)
    audit["encoder_parameter_tensor_count"] = len(list(model.encoder.parameters()))
    audit["epoch_order_sha256"] = {str(r["epoch"]): r["sample_order_sha256"] for r in refit_history}
    if model.encoder_trainable and received != set(dict(model.encoder.named_parameters())):
        raise AssertionError("Not all encoder parameter tensors received nonzero gradients")
    payload = {**identity, "initialization": initial, "audit": audit, "best_epoch": best_epoch,
               "train_pos_weight": train_weight, "refit_pos_weight": weight,
               "train_samples_sha256": sample_digest(train_examples),
               "val_samples_sha256": sample_digest(val_examples),
               "source_samples_sha256": sample_digest(source_examples),
               "test_samples_sha256": sample_digest(test_examples),
               "refit_reset_verified": not smoke,
               "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "final_model_sha256": tensor_state_hash(model.state_dict())}
    checkpoint = output / "checkpoint.pt"
    torch.save(payload, checkpoint)
    # All validation scoring uses labels only after inference; outer test is never
    # scored by smoke or the selection/refit stage.
    _, validation = score(model, bank, val_examples, task, device, protocol["batch_size"][task])
    reload_examples = val_examples[:protocol["batch_size"][task]]
    first = predict(model, bank, reload_examples, task, device, protocol["batch_size"][task])
    del optimizer, model, before
    reloaded, _ = restore_checkpoint(checkpoint, identity, device, **kwargs)
    second = predict(reloaded, bank, reload_examples, task, device, protocol["batch_size"][task])
    error = max(abs(a["logit"] - b["logit"]) for a, b in zip(first, second))
    if error != 0.0:
        raise AssertionError(f"Checkpoint reload changed logits: {error}")
    report = {**identity, "status": "passed" if smoke else "complete", "initialization": initial,
              "audit": audit, "best_epoch": best_epoch, "validation": validation,
              "train_pos_weight": train_weight, "refit_pos_weight": weight,
              "reload_max_abs_error": error, "refit_reset_verified": not smoke,
              "history": refit_history}
    write_json(output / "refit_history.json", refit_history)
    write_json(output / ("smoke_report.json" if smoke else "report.json"), report)
    return report
