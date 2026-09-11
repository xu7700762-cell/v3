from pathlib import Path
import importlib.util
import itertools

import pytest
import torch

from test_ssl_ablation import dataset, FakeEncoder
from vestibular_fusion.training import ssl_fivefold as full
from vestibular_fusion.ssl_protocol import split_identity, VARIANTS, TASKS, DATASETS
from vestibular_fusion.evaluation.io import read_json
from vestibular_fusion.evaluation.anchor_pilot import load_verified


def setup(bank, fold, task="state", variant="A2", condition="C0", smoke=False):
    parts = full.full_partitions(bank, fold, task)
    p = full.experiment_protocol()
    p.update(max_epochs=1, batch_size={"state": 7, "severity": 2}, microbatch_size={"state": 2, "severity": 1})
    identity = {"checkpoint_schema": full.SCHEMA, "training_seed": 2001, "condition": condition,
                "task": task, "variant": variant, "dataset": "monifeixing", "fold_id": "fold_2",
                "smoke": smoke, "protocol": p, "binding": {}, "official_pretrain_sha256": "official",
                "split": split_identity(fold), **parts[-1]}
    return parts[:-1], identity


def test_fractional_refit_budget():
    assert full.refit_steps(1, 33, 48) == 2
    assert full.refit_steps(54, 33, 48) == 79
    assert full.refit_steps(1980, 33, 48) == 2880
    with pytest.raises(ValueError):
        full.refit_steps(0, 33, 48)


@pytest.mark.parametrize("task", ["state", "severity"])
@pytest.mark.parametrize("variant", ["A1", "A2"])
@pytest.mark.parametrize("condition", ["C0", "C1"])
def test_full_fold_refit_outer_reload_and_compaction(dataset, tmp_path, task, variant, condition):
    bank, fold = dataset
    parts, identity = setup(bank, fold, task, variant, condition)
    job = tmp_path / condition / task / variant / "monifeixing/fold_2"
    report = full.run_fold(job, tmp_path, identity, bank, parts, torch.device("cpu"), encoder_factory=FakeEncoder)
    assert report["status"] == "complete" and report["refit_reset_verified"]
    selected = read_json(job / "selection/report.json")
    refit = read_json(job / "refit/report.json")
    assert selected["initialization"] == refit["initialization"]
    assert refit["source_pos_weight"] == full.class_weight(parts[2])
    assert report["refit_steps"] == full.refit_steps(selected["best_global_step"], selected["steps_per_epoch"], refit["source_steps_per_epoch"])
    assert not (job / "selection/best.pt").exists()
    assert not (job / "refit/boundary.pt").exists()
    raw = load_verified(job / "refit/final.pt")
    assert ("encoder_reference" in raw) == (variant == "A1")
    model, payload = full.restore_final(job / "refit/final.pt", tmp_path, identity, torch.device("cpu"), FakeEncoder)
    assert payload["role"] == "source_refit"
    assert model.encoder_trainable == (variant == "A2")
    _, rows = full.verify_job(job, tmp_path, identity)
    assert set(r["subject_id"] for r in rows) == set(fold.test_subjects)
    assert full.run_fold(job, tmp_path, identity, bank, parts, torch.device("cpu"), resume=True, encoder_factory=FakeEncoder) == report
    with pytest.raises(ValueError, match="partition"):
        full.evaluate_outer(job, tmp_path, identity, bank, parts[2], parts[5], torch.device("cpu"), FakeEncoder)


def test_refit_resume_reproduces_uninterrupted_model(dataset, tmp_path):
    bank, fold = dataset
    parts, identity = setup(bank, fold, condition="C1")
    train, val, source, _, anchors, _ = parts
    selection = full.train_pilot(tmp_path / "selection", identity, bank, train, val, anchors, torch.device("cpu"),
                                encoder_factory=FakeEncoder, checkpoint_schema=full.SCHEMA)
    # Force a known fixed refit budget spanning two complete source epochs.
    selection["best_global_step"] = selection["steps_per_epoch"] * 2
    def run(path, **kwargs):
        return full.run_refit(path, tmp_path, identity, selection, bank, source, anchors, torch.device("cpu"),
                              encoder_factory=FakeEncoder, **kwargs)
    expected = run(tmp_path / "whole")
    n = expected["source_steps_per_epoch"]
    def stop(step):
        if step == n + 1:
            raise InterruptedError("injected")
    with pytest.raises(InterruptedError):
        run(tmp_path / "interrupted", step_callback=stop)
    resumed = run(tmp_path / "interrupted", resume=True)
    a = full.final_payload(tmp_path / "whole/final.pt", tmp_path)
    b = full.final_payload(tmp_path / "interrupted/final.pt", tmp_path)
    assert a["model_sha256"] == b["model_sha256"]
    assert resumed["refit_steps"] == expected["refit_steps"]


def test_shared_frozen_encoder_reused_and_tampering_rejected(dataset, tmp_path):
    bank, fold = dataset
    for condition in ("C0", "C1"):
        parts, identity = setup(bank, fold, variant="A1", condition=condition)
        full.run_fold(tmp_path / condition, tmp_path, identity, bank, parts, torch.device("cpu"), encoder_factory=FakeEncoder)
    shared = list((tmp_path / "shared_encoders").glob("*.pt"))
    assert len(shared) == 1
    with shared[0].open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        full.final_payload(tmp_path / "C1/refit/final.pt", tmp_path)


def test_smoke_cannot_score_outer_or_complete_fivefold(dataset, tmp_path):
    bank, fold = dataset
    parts, identity = setup(bank, fold, smoke=True)
    result = full.run_fold(tmp_path / "smoke", tmp_path, identity, bank, parts, torch.device("cpu"), encoder_factory=FakeEncoder)
    assert result["status"] == "smoke_passed" and result["refit"]["refit_steps"] >= 2
    assert not (tmp_path / "smoke/evaluation").exists()
    with pytest.raises(ValueError, match="Smoke"):
        full.evaluate_outer(tmp_path / "smoke", tmp_path, identity, bank, parts[3], parts[5], torch.device("cpu"), FakeEncoder)
    with pytest.raises(ValueError, match="120"):
        full.summarize_condition(tmp_path, "C0", [identity])


def test_complete_fivefold_pooled_means_pairing_and_comparison(tmp_path, monkeypatch):
    identities = {}
    def verify(job, root, identity):
        f = identity["fold_id"]
        rows = [{"sample_id": f"{f}/s{i}", "subject_id": f"{f}/s{i}", "session": "task", "window_indices": "0",
                 "y_true": i, "y_pred": i, "score": 0.1 if i == 0 else 0.9, "logit": -2.0 if i == 0 else 2.0}
                for i in (0, 1)]
        return {"scores": full.metrics_from_rows(rows), "initialization": {"head_sha256": "head", "parameter_shapes": {},
            "encoder_sha256": str(VARIANTS[identity["variant"]]["pretrained"])}, "selection_order_sha256": f,
            "refit_order_sha256": f, "selection_best_step": 1, "selection_best_epoch": 1, "refit_steps": 2}, rows
    monkeypatch.setattr(full, "verify_job", verify)
    for c in ("C0", "C1"):
        values = [{"task": t, "variant": v, "dataset": d, "fold_id": f"fold_{f}", "condition": c,
                   "smoke": False, "checkpoint_schema": full.SCHEMA, "protocol": full.experiment_protocol(), "binding": {}}
                  for t, v, d, f in itertools.product(TASKS, VARIANTS, DATASETS, range(1, 6))]
        identities[c] = values
        result = full.summarize_condition(tmp_path, c, values)
        assert result["jobs"] == 120 and len(result["datasets"]) == 24
        assert all(r["accuracy"] == 1 and r["accuracy_fold_std"] == 0 for r in result["datasets"])
    assert full.compare_conditions(tmp_path)["jobs"] == 240
    with pytest.raises(ValueError, match="120"):
        full.summarize_condition(tmp_path, "C0", identities["C0"][:-1])


def test_cli_orders_all_c0_before_c1_and_uses_project_output():
    path = Path(__file__).resolve().parents[1] / "scripts/run_femba_ssl_fivefold.py"
    spec = importlib.util.spec_from_file_location("full_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    plan = cli.make_plan(cli.parser().parse_args([]))
    assert plan["job_count"] == 240
    assert all(j["condition"] == "C0" for j in plan["jobs"][:120])
    assert all(j["condition"] == "C1" for j in plan["jobs"][120:])
    assert Path(plan["root"]).is_relative_to(path.parents[1] / "outputs")
    with pytest.raises(ValueError, match="precede"):
        cli.make_plan(cli.parser().parse_args(["--conditions", "C1", "C0"]))
