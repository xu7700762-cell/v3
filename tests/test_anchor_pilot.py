from copy import deepcopy
from pathlib import Path
import importlib.util

import pytest
import torch

from test_ssl_ablation import dataset, FakeEncoder
from vestibular_fusion.anchor_protocol import SCHEMA, experiment_protocol
from vestibular_fusion.model import linear_probe
from vestibular_fusion.model.anchor_probe import FEMBAAnchorProbe, build_anchor_probe
from vestibular_fusion.training.anchor_data import pilot_partitions, batch_inputs
from vestibular_fusion.training.anchor_pilot import train_pilot, optimizer_step, should_validate, is_better
from vestibular_fusion.training.ssl_ablation import optimizer_for, snapshot
from vestibular_fusion.ssl_protocol import split_identity, VARIANTS
from vestibular_fusion.evaluation.anchor_pilot import (evaluate, restore_checkpoint, contrasts,
                                                     summarize, load_verified, verify_result)
from vestibular_fusion.evaluation.io import read_json


def setup(bank, fold, task="state", variant="A2", condition="C1", smoke=False):
    train, val, anchors, samples = pilot_partitions(bank, fold, task)
    protocol = experiment_protocol()
    protocol.update(max_epochs=2, batch_size={"state": 7, "severity": 2},
                    microbatch_size={"state": 2, "severity": 1})
    identity = {"checkpoint_schema": SCHEMA, "task": task, "variant": variant, "condition": condition,
                "dataset": "monifeixing", "fold_id": "fold_1", "training_seed": 2001,
                "smoke": smoke, "protocol": protocol, "split": split_identity(fold),
                "binding": {}, "official_pretrain_sha256": "official", **samples}
    return train, val, anchors, identity


@pytest.mark.parametrize("task", ["state", "severity"])
def test_centering_hand_computation_and_both_branch_gradients(dataset, task):
    bank, fold = dataset
    train, _, anchors, _ = setup(bank, fold, task)
    examples = [train[0], train[-1], train[0]]
    inputs, _ = batch_inputs(bank, examples, task, "cpu", "C1", anchors)
    x, a, mapping = inputs
    assert len(a) == 2 and mapping.tolist() == [0, 1, 0]
    model = FEMBAAnchorProbe(FakeEncoder(), encoder_trainable=True, seed=2001, condition="C1")
    x.requires_grad_()
    a.requires_grad_()
    z = model.encoder.forward_tokens(x.reshape(-1, 30, 1280)).mean(1)
    if task == "severity":
        z = z.reshape(3, 11, 525).mean(1)
    center = model.encoder.forward_tokens(a.reshape(-1, 30, 1280)).mean(1).reshape(2, 4, 525).mean(1)
    expected = model.head(z - center[mapping]).squeeze(-1)
    actual = model(x, a, mapping)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert x.grad.abs().sum() > 0 and a.grad.abs().sum() > 0
    assert model.branch_gradients == {"target": True, "anchor": True}


def test_eight_conditions_pair_initialization_and_loading(monkeypatch):
    def load(encoder, path):
        with torch.no_grad():
            encoder.scale.add_(2)
        return {"loaded_keys": 83, "missing_keys": [], "unexpected_keys": [], "skipped_keys": []}
    monkeypatch.setattr(linear_probe, "sha256_file", lambda _: "official")
    monkeypatch.setattr(linear_probe, "load_pretrained_checkpoint", load)
    infos = {}
    for c in ("C0", "C1"):
        for v in VARIANTS:
            _, infos[c, v] = build_anchor_probe(v, 2001, torch.device("cpu"), condition=c,
                checkpoint_path=Path("x"), expected_sha256="official", encoder_factory=FakeEncoder)
    assert len({i["head_sha256"] for i in infos.values()}) == 1
    assert all(i["head_parameters"] == 526 for i in infos.values())
    assert all(infos["C0", v] == infos["C1", v] for v in VARIANTS)
    assert infos["C0", "A1"] == infos["C0", "A2"]
    assert infos["C1", "A3"] == infos["C1", "A4"]
    def forbidden(*args):
        raise AssertionError("Random group accessed checkpoint")
    monkeypatch.setattr(linear_probe, "sha256_file", forbidden)
    for c in ("C0", "C1"):
        build_anchor_probe("A1", 2001, torch.device("cpu"), condition=c, encoder_factory=FakeEncoder)


def test_bad_checkpoint_and_incomplete_loading_rejected(monkeypatch):
    monkeypatch.setattr(linear_probe, "sha256_file", lambda _: "bad")
    with pytest.raises(RuntimeError, match="SHA-256"):
        build_anchor_probe("A3", 2001, torch.device("cpu"), condition="C1", checkpoint_path=Path("x"),
                           expected_sha256="official", encoder_factory=FakeEncoder)
    monkeypatch.setattr(linear_probe, "sha256_file", lambda _: "official")
    monkeypatch.setattr(linear_probe, "load_pretrained_checkpoint", lambda *a: {
        "loaded_keys": 82, "missing_keys": [], "unexpected_keys": [], "skipped_keys": []})
    with pytest.raises(RuntimeError, match="Incomplete"):
        build_anchor_probe("A4", 2001, torch.device("cpu"), condition="C1", checkpoint_path=Path("x"),
                           expected_sha256="official", encoder_factory=FakeEncoder)


@pytest.mark.parametrize("condition", ["C0", "C1"])
@pytest.mark.parametrize("trainable", [False, True])
def test_freeze_and_optimizer_and_microbatch_equivalence(dataset, condition, trainable):
    bank, fold = dataset
    train, _, anchors, identity = setup(bank, fold)
    p = identity["protocol"]
    one = FEMBAAnchorProbe(FakeEncoder(), encoder_trainable=trainable, seed=2001, condition=condition)
    two = deepcopy(one)
    one.train()
    assert one.encoder.training == trainable
    old = snapshot(one)
    optimizer = optimizer_for(one, p)
    assert {id(x) for g in optimizer.param_groups for x in g["params"]} == {id(x) for x in one.parameters() if x.requires_grad}
    a = optimizer_step(one, optimizer, bank, train[:5], "state", torch.device("cpu"), p, anchors, 0.5)
    full = {**p, "microbatch_size": {"state": 5}}
    b = optimizer_step(two, optimizer_for(two, p), bank, train[:5], "state", torch.device("cpu"), full, anchors, 0.5)
    assert a["loss"] == pytest.approx(b["loss"], abs=2e-7)
    for x, y in zip(one.parameters(), two.parameters()):
        torch.testing.assert_close(x, y, rtol=1e-6, atol=1e-7)
    if not trainable:
        assert all(torch.equal(v, one.state_dict()[k]) for k, v in old.items() if k.startswith("encoder."))
        assert not any(one.branch_gradients.values())
    else:
        assert a["encoder_gradient_names"] == ["scale"]
        assert not torch.equal(old["encoder.scale"], one.encoder.scale)


def test_partitions_reference_validation_and_no_test_access(dataset):
    bank, fold = dataset
    # A pilot must not even need the outer-test signals.
    del bank.records["s4"]
    train, val, anchors, _ = setup(bank, fold)
    assert set(anchors) == {"s1", "s2", "s3"}
    for e in train + val:
        assert not set(e.indices) & set(anchors[e.subject_id]["indices"])
    altered = deepcopy(anchors)
    altered["s1"]["reference_session"] = "task"
    with pytest.raises(ValueError, match="reference anchors"):
        batch_inputs(bank, train[:2], "state", "cpu", "C1", altered)
    inputs, count = batch_inputs(bank, train[:2], "state", "cpu", "C0", {})
    assert inputs[1:] == (None, None) and count == 2


def test_early_selection_and_tie():
    assert [i for i in range(1, 11) if should_validate(i, 10)] == [1, 3, 6, 9, 10]
    assert should_validate(1, 1)
    assert is_better(0.7, None)
    assert not is_better(0.7, {"validation": {"loss": 0.7}})
    assert is_better(0.6, {"validation": {"loss": 0.7}})


def test_interrupted_checkpoint_commit_preserves_previous_boundary(tmp_path, monkeypatch):
    from vestibular_fusion.training import anchor_pilot as training
    path = tmp_path / "boundary.pt"
    training.save_checkpoint(path, {"global_step": 6, "weights": torch.ones(2)})
    real_replace = training.os.replace
    def interrupt_manifest(source, destination):
        if Path(source).name == "boundary.tmp.sha256.json" and Path(destination).name == "boundary.sha256.json":
            raise InterruptedError("interrupted before manifest commit")
        return real_replace(source, destination)
    monkeypatch.setattr(training.os, "replace", interrupt_manifest)
    with pytest.raises(InterruptedError):
        training.save_checkpoint(path, {"global_step": 12, "weights": torch.zeros(2)})
    with pytest.raises(RuntimeError, match="integrity"):
        load_verified(path)
    monkeypatch.setattr(training.os, "replace", real_replace)
    assert load_verified(path, recover=True)["global_step"] == 6
    assert load_verified(path)["global_step"] == 6


@pytest.mark.parametrize("task", ["state", "severity"])
def test_train_evaluate_reload_and_refuse_test(dataset, tmp_path, task):
    bank, fold = dataset
    train, val, anchors, identity = setup(bank, fold, task, smoke=True)
    job = tmp_path / "job"
    report = train_pilot(job / "training", identity, bank, train, val, anchors, torch.device("cpu"), encoder_factory=FakeEncoder)
    assert report["global_step"] == 2
    assert report["best_global_step"] <= 2 and report["best_epoch"] < 20
    assert report["reload_max_abs_error"] == 0
    evaluate(job / "training/best.pt", job / "evaluation", identity, bank, val, anchors,
             torch.device("cpu"), FakeEncoder)
    verify_result(job, identity)
    with pytest.raises(ValueError, match="source-val"):
        evaluate(job / "training/best.pt", job / "bad", identity, bank, train, anchors, torch.device("cpu"), FakeEncoder)
    wrong = {**identity, "condition": "C0"}
    with pytest.raises(RuntimeError, match="mismatch"):
        restore_checkpoint(job / "training/best.pt", wrong, torch.device("cpu"), FakeEncoder)
    with pytest.raises(FileExistsError):
        train_pilot(job / "training", identity, bank, train, val, anchors, torch.device("cpu"), encoder_factory=FakeEncoder)


def test_resume_rolls_back_partial_epoch_best_and_history(dataset, tmp_path):
    bank, fold = dataset
    train, val, anchors, identity = setup(bank, fold)
    def run(folder, **kwargs):
        return train_pilot(folder, identity, bank, train, val, anchors, torch.device("cpu"), encoder_factory=FakeEncoder, **kwargs)
    uninterrupted = run(tmp_path / "full")
    def interrupt(step):
        if step == 7:
            raise InterruptedError("intentional test interruption")
    with pytest.raises(InterruptedError):
        run(tmp_path / "resumed", step_callback=interrupt)
    boundary = load_verified(tmp_path / "resumed/boundary.pt")
    assert boundary["global_step"] == 6
    resumed = run(tmp_path / "resumed", resume=True)
    assert resumed["resume_events"][0]["restored_step"] == 6
    for name in ("best", "last"):
        a = load_verified(tmp_path / f"full/{name}.pt")
        b = load_verified(tmp_path / f"resumed/{name}.pt")
        assert a["model_sha256"] == b["model_sha256"]
        assert a["global_step"] == b["global_step"]
    assert resumed["best_validation"] == uninterrupted["best_validation"]
    history = read_json(tmp_path / "resumed/history.json")["checks"]
    assert [r["global_step"] for r in history] == list(range(1, 13))


def test_contrasts_do_not_confuse_random_decline_with_pretrained_improvement():
    values = {(c, v): 0.6 for c in ("C0", "C1") for v in VARIANTS}
    values["C1", "A1"] = 0.4
    result = contrasts(values)
    assert result["interaction_pp"]["frozen"] == pytest.approx(20)
    assert not result["improvement_and_amplification"]["frozen"]
    values["C1", "A3"] = 0.7
    assert contrasts(values)["improvement_and_amplification"]["frozen"]


def test_cli_matrix_and_partial_summary_rejected(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/run_femba_ssl_anchor_pilot.py"
    spec = importlib.util.spec_from_file_location("anchor_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    plan = cli.make_plan(cli.parser().parse_args([]))
    assert plan["job_count"] == 48 and not plan["outer_test_scoring"]
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["--folds", "2"])
    with pytest.raises(ValueError):
        cli.make_plan(cli.parser().parse_args(["--stage", "summarize", "--conditions", "C1"]))
    with pytest.raises(ValueError, match="48"):
        summarize(tmp_path, [])


def test_complete_summary_checks_all_conditions_and_artifacts(dataset, tmp_path, monkeypatch):
    bank, fold = dataset
    monkeypatch.setattr(linear_probe, "sha256_file", lambda _: "official")
    def load(encoder, path):
        with torch.no_grad():
            encoder.scale.add_(1)
        return {"loaded_keys": 83, "missing_keys": [], "unexpected_keys": [], "skipped_keys": []}
    monkeypatch.setattr(linear_probe, "load_pretrained_checkpoint", load)
    identities = []
    for task in ("state", "severity"):
        for d in ("monifeixing", "vrq", "city"):
            for c in ("C0", "C1"):
                for v in VARIANTS:
                    train, val, anchors, identity = setup(bank, fold, task, v, c)
                    identity["dataset"] = d
                    identity["protocol"]["max_epochs"] = 1
                    identity["protocol"]["batch_size"] = {"state": 40, "severity": 2}
                    identities.append(identity)
                    job = tmp_path / task / c / v / d / "fold_1"
                    train_pilot(job / "training", identity, bank, train, val, anchors, torch.device("cpu"),
                                checkpoint_path=Path("official"), encoder_factory=FakeEncoder)
                    evaluate(job / "training/best.pt", job / "evaluation", identity, bank, val, anchors,
                             torch.device("cpu"), FakeEncoder)
    report = summarize(tmp_path, identities)
    assert report["jobs"] == 48 and report["pilot_completed"] and not report["full_fivefold_completed"]
    assert len(report["metrics"]) == 48
    first = identities[0]
    job = tmp_path / first["task"] / first["condition"] / first["variant"] / first["dataset"] / "fold_1"
    bad = {**first, "condition": "C1"}
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_result(job, bad)
    with (job / "evaluation/predictions.csv").open("a") as handle:
        handle.write("tampered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_result(job, first)
