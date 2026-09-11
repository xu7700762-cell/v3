from copy import deepcopy
from dataclasses import replace
import importlib.util
import itertools
from pathlib import Path

import pytest
import torch

from test_ssl_ablation import FakeEncoder, dataset
from vestibular_fusion.model import linear_probe
from vestibular_fusion.model.anchor_probe import build_anchor_probe
from vestibular_fusion.model.c0_heads import FEMBAC0HeadProbe, head_factory, HEAD_PARAMETERS
from vestibular_fusion.training import c0_head_comparison as comparison
from vestibular_fusion.training import ssl_fivefold as full
from vestibular_fusion.training.anchor_pilot import train_pilot
from vestibular_fusion.training.ssl_ablation import optimizer_for, snapshot
from vestibular_fusion.evaluation.anchor_pilot import load_verified, score
from vestibular_fusion.evaluation.c0_head_comparison import summary_from_folds
from vestibular_fusion.evaluation.io import read_json
from vestibular_fusion.ssl_protocol import split_identity


@pytest.fixture
def pretrained(monkeypatch):
    monkeypatch.setattr(linear_probe, "sha256_file", lambda _: "official")
    def load(encoder, _):
        with torch.no_grad():
            encoder.scale.add_(0.2)
        return {"loaded_keys": 83, "missing_keys": [], "unexpected_keys": [], "skipped_keys": []}
    monkeypatch.setattr(linear_probe, "load_pretrained_checkpoint", load)


def setup(bank, fold, head="mlp", variant="A4", task="state", smoke=False):
    *parts, samples = full.full_partitions(bank, fold, task)
    protocol = comparison.experiment_protocol()
    protocol.update(max_epochs=2, batch_size={"state": 7, "severity": 2}, microbatch_size={"state": 2, "severity": 1})
    identity = {"checkpoint_schema": comparison.SCHEMA, "training_seed": 2001, "condition": "C0", "head": head,
        "task": task, "variant": variant, "dataset": "vrq", "fold_id": "fold_1", "smoke": smoke,
        "protocol": protocol, "binding": {}, "split": split_identity(fold), **samples,
        "official_pretrain_sha256": "official", "encoder_trainable": variant == "A4", "environment": {}}
    return parts, identity


def test_parameter_counts_output_initialization_and_random_streams(pretrained):
    initial = {}
    for h, v in itertools.product(comparison.HEADS, comparison.VARIANTS):
        before = torch.get_rng_state().clone()
        model, info = build_anchor_probe(v, 2001, torch.device("cpu"), condition="C0", model_factory=head_factory(h),
            encoder_factory=FakeEncoder, checkpoint_path=Path("official"), expected_sha256="official")
        assert torch.equal(before, torch.get_rng_state())
        assert info["head_parameters"] == HEAD_PARAMETERS[h]
        assert all(p.requires_grad for p in model.head.parameters())
        optimized = {id(p) for g in optimizer_for(model, comparison.experiment_protocol()).param_groups for p in g["params"]}
        assert optimized == {id(p) for p in model.parameters() if p.requires_grad}
        initial[h, v] = info
    assert len({i["encoder_sha256"] for i in initial.values()}) == 1
    assert len({i["output_layer_sha256"] for i in initial.values()}) == 1
    for h in comparison.HEADS:
        assert initial[h, "A3"] == initial[h, "A4"]


@pytest.mark.parametrize("head", comparison.HEADS)
@pytest.mark.parametrize("task", comparison.TASKS)
def test_pool_before_mapping_and_no_reference_inputs(head, task):
    model = FEMBAC0HeadProbe(FakeEncoder(), encoder_trainable=True, seed=2001, condition="C0", head_kind=head)
    x = torch.randn((2, 11, 30, 1280) if task == "severity" else (2, 30, 1280))
    z = model.encoder.forward_tokens(x.reshape(-1, 30, 1280)).float().mean(1)
    if task == "severity":
        z = z.reshape(2, 11, 525).mean(1)
    torch.testing.assert_close(model(x), model.head(z).squeeze(-1), rtol=0, atol=0)
    with pytest.raises(ValueError, match="anchors"):
        model(x, torch.zeros(1, 4, 30, 1280), torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError, match="C0"):
        FEMBAC0HeadProbe(FakeEncoder(), encoder_trainable=True, seed=2001, condition="C1", head_kind=head)


def test_fractional_and_dog_gradients_and_extreme_inputs():
    model = FEMBAC0HeadProbe(FakeEncoder(), encoder_trainable=False, seed=2001, condition="C0", head_kind="fractional_dog_polykan")
    model.head(torch.randn(4, 525)).square().sum().backward()
    for name in ("fractional_order_logit", "dog_mix_logits"):
        assert getattr(model.head[1], name).grad.abs().sum() > 0
    assert torch.isfinite(model.head(torch.randn(3, 525) * 1e6)).all()


@pytest.mark.parametrize("head", comparison.HEADS)
@pytest.mark.parametrize("variant", comparison.VARIANTS)
@pytest.mark.parametrize("task", comparison.TASKS)
def test_full_workflow_update_reload_and_stage_isolation(dataset, pretrained, tmp_path, head, variant, task):
    bank, fold = dataset
    parts, identity = setup(bank, fold, head, variant, task)
    job = tmp_path / "job"
    trained = comparison.run_job(job, tmp_path, identity, bank, parts, torch.device("cpu"), stage="train",
                                checkpoint_path=Path("official"), encoder_factory=FakeEncoder)
    assert trained["status"] == "trained" and not (job / "evaluation").exists()
    selected = read_json(job / "selection/report.json")
    assert selected["audit"]["head_delta_l2"] > 0
    assert (selected["audit"]["encoder_delta_l2"] > 0) == (variant == "A4")
    if variant == "A3":
        assert selected["audit"]["encoder_changed_tensors"] == []
    report = comparison.run_job(job, tmp_path, identity, bank, parts, torch.device("cpu"), stage="evaluate",
                               encoder_factory=FakeEncoder)
    assert report["status"] == "complete"
    assert comparison.run_job(job, tmp_path, identity, bank, parts, torch.device("cpu"), stage="evaluate",
                              encoder_factory=FakeEncoder) == report
    model, payload = full.restore_final(job / "refit/final.pt", tmp_path, identity, torch.device("cpu"), FakeEncoder,
                                       checkpoint_schema=comparison.SCHEMA, model_factory=head_factory(head))
    model.train()
    assert model.encoder.training == (variant == "A4")
    rows, _ = score(model, bank, parts[3], task, torch.device("cpu"), 1, {})
    altered = [replace(e, label=1-e.label) for e in parts[3]]
    changed, _ = score(model, bank, altered, task, torch.device("cpu"), 1, {})
    assert [r["logit"] for r in rows] == [r["logit"] for r in changed]
    with pytest.raises(FileExistsError):
        comparison.run_job(job, tmp_path, identity, bank, parts, torch.device("cpu"), encoder_factory=FakeEncoder)
    wrong = {**identity, "head": "mlp" if head != "mlp" else "fractional_dog_polykan"}
    with pytest.raises(RuntimeError, match="mismatch"):
        full.restore_final(job / "refit/final.pt", tmp_path, wrong, torch.device("cpu"), FakeEncoder,
                           checkpoint_schema=comparison.SCHEMA, model_factory=head_factory(wrong["head"]))


@pytest.mark.parametrize("head", comparison.HEADS)
def test_selection_and_refit_boundary_resume_matches_uninterrupted(dataset, pretrained, tmp_path, head):
    bank, fold = dataset
    parts, identity = setup(bank, fold, head)
    train, val, source, _, anchors, _ = parts
    options = dict(checkpoint_path=Path("official"), encoder_factory=FakeEncoder,
                   model_factory=head_factory(head), checkpoint_schema=comparison.SCHEMA)
    def selection(path, **kw):
        return train_pilot(path, identity, bank, train, val, anchors, torch.device("cpu"), **options, **kw)
    whole = selection(tmp_path / "whole")
    def stop(step):
        if step == 7:
            raise InterruptedError("injected partial epoch")
    with pytest.raises(InterruptedError):
        selection(tmp_path / "interrupted", step_callback=stop)
    resumed = selection(tmp_path / "interrupted", resume=True)
    for name in ("best", "last"):
        a, b = (load_verified(tmp_path / folder / f"{name}.pt") for folder in ("whole", "interrupted"))
        assert a["model_sha256"] == b["model_sha256"] and a["global_step"] == b["global_step"]
    assert resumed["best_validation"] == whole["best_validation"]
    whole["best_global_step"] = whole["steps_per_epoch"] * 2
    def refit(path, **kw):
        return full.run_refit(path, tmp_path, identity, whole, bank, source, anchors, torch.device("cpu"), **options, **kw)
    complete = refit(tmp_path / "refit_whole")
    def stop_refit(step):
        if step == complete["source_steps_per_epoch"] + 1:
            raise InterruptedError("injected refit")
    with pytest.raises(InterruptedError):
        refit(tmp_path / "refit_interrupted", step_callback=stop_refit)
    refit(tmp_path / "refit_interrupted", resume=True)
    assert full.final_payload(tmp_path / "refit_whole/final.pt", tmp_path)["model_sha256"] == full.final_payload(
        tmp_path / "refit_interrupted/final.pt", tmp_path)["model_sha256"]


def test_smoke_never_scores_outer(dataset, pretrained, tmp_path, monkeypatch):
    bank, fold = dataset
    parts, identity = setup(bank, fold, smoke=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("Smoke reached outer-test")
    monkeypatch.setattr(full, "evaluate_outer", forbidden)
    result = comparison.run_job(tmp_path / "smoke", tmp_path, identity, bank, parts, torch.device("cpu"),
                                checkpoint_path=Path("official"), encoder_factory=FakeEncoder)
    assert result["selection"]["global_step"] == 2 and result["refit"]["refit_steps"] >= 2
    assert not result["outer_test_scored"]


def test_summary_completeness_pairing_macros_and_negative_results():
    folds = {}
    for h, t, v, d, f in itertools.product(("linear", *comparison.HEADS), comparison.TASKS,
                                         comparison.VARIANTS, comparison.DATASETS, comparison.FOLDS):
        rows = [{"sample_id": f"{f}/{i}", "subject_id": f"{f}/{i}", "session": "task", "window_indices": "0",
                 "y_true": i, "y_pred": i, "score": 0.1 if i == 0 else 0.9, "logit": -2.0 if i == 0 else 2.0} for i in (0, 1)]
        if h == "fractional_dog_polykan":
            for row in rows:
                row.update(y_pred=1-row["y_pred"], score=1-row["score"], logit=-row["logit"])
        folds[h, t, v, d, f] = rows
    result = summary_from_folds(folds)
    assert len(result["folds"]) == 120 and len(result["datasets"]) == 24
    gains = [x["difference"] for x in result["comparisons"] if x["comparison"] == "fractional_dog_polykan-mlp" and x["metric"] == "balanced_accuracy"]
    assert all(x == -100 for x in gains)
    missing = dict(folds)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="80"):
        summary_from_folds(missing)
    altered = deepcopy(folds)
    altered[next(iter(altered))][0]["y_true"] = 1
    with pytest.raises(RuntimeError, match="labels"):
        summary_from_folds(altered)


def test_cli_limits_matrix_storage_and_unknown_seeds():
    path = Path(__file__).resolve().parents[1] / "scripts/run_femba_c0_head_comparison.py"
    spec = importlib.util.spec_from_file_location("head_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.make_plan(cli.parser().parse_args([]))["job_count"] == 80
    assert cli.make_plan(cli.parser().parse_args(["--smoke"]))["job_count"] == 16
    with pytest.raises(ValueError, match="80"):
        cli.make_plan(cli.parser().parse_args(["--stage", "summarize", "--heads", "mlp"]))
    with pytest.raises(ValueError, match="D:/"):
        cli.make_plan(cli.parser().parse_args(["--output-root", "/tmp/not_on_d"]))
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["--seed", "3001"])


def test_baseline_protocol_changes_are_rejected_before_training(dataset):
    bank, fold = dataset
    parts, identity = setup(bank, fold)
    key = tuple(identity[k] for k in ("task", "variant", "dataset", "fold_id"))
    previous = deepcopy(identity)
    previous["protocol"]["head_lr"] = 0.123
    baseline = {"jobs": {key: {"identity": previous}}}
    with pytest.raises(RuntimeError, match="head_lr"):
        comparison.check_baseline_pair(identity, parts, baseline)
