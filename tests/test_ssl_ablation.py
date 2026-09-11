from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vestibular_fusion.config import load_config
from vestibular_fusion.data.types import SubjectRecord
from vestibular_fusion.data.anchors import anchor_indices, task_indices
from vestibular_fusion.model import linear_probe as probe
from vestibular_fusion.ssl_protocol import (SCHEMA, VARIANTS, FOLDS, DATASETS, experiment_protocol,
    protocol_binding, split_identity, validate_metadata)
from vestibular_fusion.training.data import FoldProtocol, SeverityExample
from vestibular_fusion.training.probe_data import (make_examples, class_weight, epoch_order,
                                                  input_tensor, sample_digest)
from vestibular_fusion.training.ssl_ablation import optimizer_for, train_epoch, train_fold, choose_epoch
from vestibular_fusion.evaluation.ssl_ablation import (predict, score, restore_checkpoint,
                                                      summarize, _row_identity, evaluate_fold)
from vestibular_fusion.evaluation.io import write_json, write_csv, sha256_file
from vestibular_fusion.evaluation.metrics import binary_metrics


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.linspace(0.5, 1.5, 525))
        self.register_buffer("calls_in_train", torch.zeros((), dtype=torch.int64))

    def forward_tokens(self, x):
        if self.training:
            self.calls_in_train.add_(1)
        value = x[:, 0].mean(-1)[:, None, None]
        positions = torch.linspace(-1, 1, 80, device=x.device)[None, :, None]
        return value * self.scale[None, None] + positions


@pytest.fixture
def dataset():
    rng = np.random.default_rng(9)
    records = {}
    for s in ("s1", "s2", "s3", "s4"):
        windows = rng.normal(size=(24, 30, 1280)).astype(np.float32)
        windows[:12] -= 0.3
        windows[12:] += 0.3
        records[s] = SubjectRecord(windows, np.array([0] * 12 + [1] * 12),
                                   ["rest"] * 12 + ["task"] * 12)
    severity = tuple(SeverityExample(s, "rest", "task", i % 2)
                     for i, s in enumerate(records))
    fold = FoldProtocol(("s1", "s2", "s3"), ("s4",), ("s1", "s2"), ("s3",),
                        severity[:3], severity[3:])
    return SimpleNamespace(records=records), fold


def identity(fold, task="state", variant="A1", smoke=False, protocol=None):
    protocol = protocol or experiment_protocol(2001)
    return {"checkpoint_schema": SCHEMA, "variant": variant, "task": task,
            "dataset": "monifeixing", "fold_id": "fold_1", "training_seed": 2001,
            "encoder_trainable": VARIANTS[variant]["encoder_trainable"], "smoke": smoke,
            "split": split_identity(fold), "protocol": protocol, "binding": {},
            "official_pretrain_sha256": "official"}


def test_random_groups_do_not_read_any_checkpoint(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Random variant tried to access pretrained weights")
    monkeypatch.setattr(probe, "sha256_file", forbidden)
    monkeypatch.setattr(probe, "load_pretrained_checkpoint", forbidden)
    a, ia = probe.build_probe("A1", 2001, torch.device("cpu"), checkpoint_path=Path("missing"),
                              encoder_factory=FakeEncoder)
    b, ib = probe.build_probe("A2", 2001, torch.device("cpu"), encoder_factory=FakeEncoder)
    assert ia == ib
    assert ia["head_parameters"] == 526
    assert ia["pretrain_load_info"] is None
    assert all(not p.requires_grad for p in a.encoder.parameters())
    assert all(p.requires_grad for p in b.encoder.parameters())


def test_four_way_shapes_heads_and_pairing(monkeypatch):
    monkeypatch.setattr(probe, "sha256_file", lambda path: "official")
    def load(encoder, path):
        with torch.no_grad():
            encoder.scale.add_(2)
        return {"loaded_keys": 83, "missing_keys": [], "unexpected_keys": [], "skipped_keys": []}
    monkeypatch.setattr(probe, "load_pretrained_checkpoint", load)
    info = [probe.build_probe(v, 2001, torch.device("cpu"), checkpoint_path=Path("official"),
                             expected_sha256="official", encoder_factory=FakeEncoder)[1]
            for v in VARIANTS]
    assert len({i["head_sha256"] for i in info}) == 1
    assert all(i["parameter_shapes"] == info[0]["parameter_shapes"] for i in info)
    assert info[0]["encoder_sha256"] == info[1]["encoder_sha256"]
    assert info[2]["encoder_sha256"] == info[3]["encoder_sha256"]
    assert info[0]["encoder_sha256"] != info[2]["encoder_sha256"]


def test_pretrained_sha_and_incomplete_load_rejected(monkeypatch):
    monkeypatch.setattr(probe, "sha256_file", lambda path: "wrong")
    with pytest.raises(RuntimeError, match="SHA-256"):
        probe.build_probe("A3", 1, torch.device("cpu"), checkpoint_path=Path("x"),
                          expected_sha256="right", encoder_factory=FakeEncoder)
    monkeypatch.setattr(probe, "sha256_file", lambda path: "right")
    monkeypatch.setattr(probe, "load_pretrained_checkpoint", lambda *args: {
        "loaded_keys": 82, "missing_keys": [], "unexpected_keys": [], "skipped_keys": []})
    with pytest.raises(RuntimeError, match="Incomplete"):
        probe.build_probe("A4", 1, torch.device("cpu"), checkpoint_path=Path("x"),
                          expected_sha256="right", encoder_factory=FakeEncoder)


@pytest.mark.parametrize("trainable", [False, True])
def test_freeze_mode_optimizer_and_real_updates(dataset, trainable):
    bank, fold = dataset
    model = probe.FEMBALinearProbe(FakeEncoder(), encoder_trainable=trainable, seed=2001)
    model.eval().train()
    assert model.encoder.training == trainable
    config = experiment_protocol(2001)
    optimizer = optimizer_for(model, config)
    ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert all((id(p) in ids) == trainable for p in model.encoder.parameters())
    before = probe.tensor_state_hash(model.encoder.state_dict())
    examples = make_examples(bank, fold, "state", fold.source_train_subjects)
    train_epoch(model, optimizer, bank, examples, "state", torch.device("cpu"), config, 1,
                class_weight(examples), max_steps=2)
    assert (before != probe.tensor_state_hash(model.encoder.state_dict())) == trainable
    assert all((p.grad is not None) == trainable for p in model.encoder.parameters())
    if not trainable:
        assert model.encoder.calls_in_train.item() == 0


def test_pooling_is_exactly_token_then_window_mean():
    model = probe.FEMBALinearProbe(FakeEncoder(), encoder_trainable=True, seed=5).eval()
    windows = torch.randn(2, 11, 30, 1280)
    pooled = model.encoder.forward_tokens(windows.flatten(0, 1)).mean(1).reshape(2, 11, 525).mean(1)
    assert torch.equal(model(windows), model.head(pooled).squeeze(-1))
    with pytest.raises(ValueError, match="11"):
        model(windows[:, :10])


def test_sampling_and_partitions(dataset):
    bank, fold = dataset
    state = make_examples(bank, fold, "state", fold.source_train_subjects)
    for s in fold.source_train_subjects:
        actual = {e.indices[0] for e in state if e.subject_id == s}
        assert actual == set(range(24)) - set(anchor_indices(bank.records[s], "rest"))
    severity = make_examples(bank, fold, "severity", fold.source_train_subjects)
    assert all(e.indices == tuple(task_indices(bank.records[e.subject_id], "task", 11)) for e in severity)
    assert input_tensor(bank, state[:2], "state", "cpu").shape == (2, 30, 1280)
    assert input_tensor(bank, severity, "severity", "cpu").shape == (2, 11, 30, 1280)
    assert class_weight(severity) == 1
    assert set(epoch_order(len(state), 2001, 1)) == set(range(len(state)))
    assert np.array_equal(epoch_order(40, 2001, 1), epoch_order(40, 2001, 1))
    assert not np.array_equal(epoch_order(40, 2001, 1), epoch_order(40, 2001, 2))
    with pytest.raises(ValueError, match="partition"):
        make_examples(bank, fold, "state", ["unknown"])


def test_test_labels_cannot_change_predictions(dataset):
    bank, fold = dataset
    model = probe.FEMBALinearProbe(FakeEncoder(), encoder_trainable=False, seed=1)
    examples = make_examples(bank, fold, "state", fold.test_subjects)
    first = predict(model, bank, examples, "state", torch.device("cpu"), 32)
    second = predict(model, bank, [replace(e, label=1-e.label) for e in examples],
                     "state", torch.device("cpu"), 32)
    assert first == second
    assert all("y_true" not in row for row in first)


def test_selection_is_minimum_eligible_bce_with_earliest_tie():
    protocol = {"min_epochs": 20}
    history = [{"epoch": e, "validation": {"loss": loss}} for e, loss in [(1, .01), (20, .5), (21, .4), (22, .4)]]
    assert choose_epoch(history, protocol) == 21


@pytest.mark.parametrize("variant", ["A1", "A2"])
def test_full_selection_refit_and_reload_on_synthetic_data(dataset, tmp_path, variant):
    bank, fold = dataset
    protocol = {**experiment_protocol(2001), "min_epochs": 2, "max_epochs": 3, "patience": 1}
    meta = identity(fold, variant=variant, protocol=protocol)
    partitions = [make_examples(bank, fold, "state", subjects) for subjects in (
        fold.source_train_subjects, fold.source_val_subjects, fold.source_subjects, fold.test_subjects)]
    report = train_fold(tmp_path / "run", meta, bank, *partitions, torch.device("cpu"),
                        encoder_factory=FakeEncoder)
    assert report["refit_reset_verified"]
    assert report["best_epoch"] >= 2
    assert report["reload_max_abs_error"] == 0
    checkpoint = tmp_path / "run" / "checkpoint.pt"
    model, payload = restore_checkpoint(checkpoint, meta, torch.device("cpu"), encoder_factory=FakeEncoder)
    assert payload["refit_pos_weight"] == class_weight(partitions[2])
    assert payload["initialization"]["head_parameters"] == 526
    result = evaluate_fold(checkpoint, tmp_path / "evaluation", meta, bank, partitions[3],
                           torch.device("cpu"), encoder_factory=FakeEncoder)
    assert result["status"] == "complete"
    assert result["test_samples_sha256"] == sample_digest(partitions[3])
    assert result["metrics"]["n_samples"] == len(partitions[3])
    with pytest.raises(ValueError, match="Smoke"):
        evaluate_fold(checkpoint, tmp_path / "not_written", {**meta, "smoke": True},
                      bank, partitions[3], torch.device("cpu"), encoder_factory=FakeEncoder)
    with pytest.raises(RuntimeError, match="metadata"):
        restore_checkpoint(checkpoint, {**meta, "variant": "A3"}, "cpu", encoder_factory=FakeEncoder)
    with pytest.raises(FileExistsError):
        train_fold(tmp_path / "run", meta, bank, *partitions, torch.device("cpu"), encoder_factory=FakeEncoder)


def test_random_only_preflight_binding_matches_pretrained_binding(tmp_path, monkeypatch):
    import vestibular_fusion.ssl_protocol as module
    monkeypatch.setattr(module, "sha256_file", lambda path: "same")
    shared = {"label": "data", "size": 1, "sha256": "a"}
    checkpoint = {"label": "pretrained FEMBA checkpoint", "size": 2, "sha256": "b"}
    protocol = experiment_protocol(2001)
    assert protocol_binding(protocol, {"files": [shared]}, tmp_path) == protocol_binding(
        protocol, {"files": [shared, checkpoint]}, tmp_path)


def test_random_preflight_does_not_check_checkpoint(monkeypatch, tmp_path):
    import vestibular_fusion.preflight as module
    from vestibular_fusion.config import DEFAULT_PROTOCOL_ROOT
    checked = []
    def fake_hash(path, expected, label):
        checked.append(label)
        return {"label": label, "size": 1, "sha256": expected}
    monkeypatch.setattr(module, "_require_hash", fake_hash)
    config = {"protocol_root": DEFAULT_PROTOCOL_ROOT, "paths": {
        k: tmp_path for k in ("monifeixing_data_root", "vrq_data_root", "city_data_root")}}
    module._check_protocol(config, require_pretrain=False)
    assert checked and "pretrained FEMBA checkpoint" not in checked


def test_random_only_configuration_omits_checkpoint(tmp_path):
    path = tmp_path / "paths.json"
    path.write_text(json.dumps({"paths": {k: "data" for k in (
        "monifeixing_data_root", "vrq_data_root", "city_data_root")}}))
    assert "pretrain_checkpoint" not in load_config(path, require_pretrain=False)["paths"]
    with pytest.raises(ValueError, match="pretrain_checkpoint"):
        load_config(path)


def test_cli_dry_plan_and_smoke_guards():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_femba_ssl_ablation.py"
    spec = importlib.util.spec_from_file_location("ssl_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.make_plan(module.parser().parse_args([]))["job_count"] == 120
    args = module.parser().parse_args(["--smoke", "--folds", "1"])
    plan = module.make_plan(args)
    assert plan["job_count"] == 24 and not plan["complete_matrix"]
    for argv in (["--smoke", "--stage", "evaluate"],
                 ["--stage", "summarize", "--folds", "1"], ["--variants", "A1", "A1"],
                 ["--resume", "--smoke"], ["--resume", "--stage", "train"]):
        with pytest.raises(ValueError):
            module.make_plan(module.parser().parse_args(argv))


def load_resume_cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_femba_ssl_ablation.py"
    spec = importlib.util.spec_from_file_location("resume_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_archives_interrupted_fold_without_touching_completed_sibling(tmp_path):
    module = load_resume_cli()
    root = tmp_path / "seed_2001"
    job = root / "state/A2/monifeixing/fold_5"
    job.mkdir(parents=True)
    (job / "interrupted.log").write_text("preserve me")
    sibling = root / "state/A2/monifeixing/fold_4"
    sibling.mkdir()
    (sibling / "finished.log").write_text("completed")
    assert module.resume_fold(job, root, {}) is None
    archived = list((tmp_path / "interrupted_attempts").rglob("interrupted.log"))
    assert len(archived) == 1 and archived[0].read_text() == "preserve me"
    assert not job.exists() and (sibling / "finished.log").read_text() == "completed"
    with pytest.raises(ValueError):
        module.resume_fold(tmp_path, root, {})


def test_resume_verifies_completed_artifacts_and_rejects_corruption(tmp_path):
    module = load_resume_cli()
    root = tmp_path / "seed_2001"
    job = root / "state/A1/monifeixing/fold_1"
    meta = {"variant": "A1", "smoke": False, "training_seed": 2001}
    training = job / "training"
    training.mkdir(parents=True)
    (training / "checkpoint.pt").write_bytes(b"valid checkpoint")
    rows = [{"subject_id": "s", "y_true": i, "y_pred": i, "score": .1 + .8*i} for i in (0, 1)]
    write_csv(job / "evaluation/predictions.csv", rows)
    tr = {**meta, "status": "complete", "initialization": {"head_sha256": "same"},
          "refit_reset_verified": True, "reload_max_abs_error": 0}
    ev = {**tr, "metrics": binary_metrics(rows),
          "checkpoint_sha256": sha256_file(training / "checkpoint.pt"),
          "predictions_sha256": sha256_file(job / "evaluation/predictions.csv")}
    write_json(training / "report.json", tr)
    write_json(job / "evaluation/report.json", ev)
    assert module.resume_fold(job, root, meta) == tr
    assert not (tmp_path / "interrupted_attempts").exists()
    (training / "checkpoint.pt").write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        module.resume_fold(job, root, meta)
    assert job.exists()


def write_summary_fixture(root, protocol):
    splits = {d: {} for d in DATASETS}
    for d in DATASETS:
        for f in FOLDS:
            splits[d][f] = {"source_subjects": ["train"], "source_train_subjects": ["train"],
                             "source_val_subjects": [], "test_subjects": [f]}
            for v in VARIANTS:
                folder = root / "state" / v / d / f
                training = folder / "training"
                training.mkdir(parents=True)
                (training / "checkpoint.pt").write_bytes(b"test-checkpoint")
                rows = [{"sample_id": f"{f}/{i}", "subject_id": f, "session": "task",
                         "window_indices": str(i), "y_true": i, "score": .2 + i*.6, "y_pred": i}
                        for i in range(2)]
                output = folder / "evaluation"
                write_csv(output / "predictions.csv", rows)
                meta = {"checkpoint_schema": SCHEMA, "status": "complete", "smoke": False,
                        "variant": v, "task": "state", "dataset": d, "fold_id": f,
                        "training_seed": 2001, "binding": {}, "protocol": protocol,
                        "encoder_trainable": VARIANTS[v]["encoder_trainable"],
                        "official_pretrain_sha256": "official", "split": splits[d][f],
                        "predictions_sha256": sha256_file(output / "predictions.csv"),
                        "checkpoint_sha256": sha256_file(training / "checkpoint.pt"),
                        "metrics": binary_metrics(rows), "audit": {"epoch_order_sha256": {"1": "order"}},
                        "initialization": {"head_sha256": "head", "parameter_shapes": {},
                                           "encoder_sha256": "pretrained" if VARIANTS[v]["pretrained"] else "random"}}
                write_json(output / "report.json", meta)
    return splits


def test_complete_summary_and_contrasts(tmp_path):
    protocol = experiment_protocol(2001)
    splits = write_summary_fixture(tmp_path, protocol)
    report = summarize(tmp_path, protocol, {}, "official", ["state"], splits)
    assert report["status"] == "complete"
    assert report["groups"]["state"]["A1"]["macro"]["accuracy"] == 1
    assert report["contrasts_percentage_points"]["state"]["A3-A1"]["macro"]["AUROC"] == 0


@pytest.mark.parametrize("corruption", ["missing_fold", "mixed_variant", "smoke", "different_samples"])
def test_summary_rejects_incomplete_or_incomparable_results(tmp_path, corruption):
    protocol = experiment_protocol(2001)
    splits = write_summary_fixture(tmp_path, protocol)
    folder = tmp_path / "state" / "A2" / "city" / "fold_5" / "evaluation"
    path = folder / "report.json"
    report = json.loads(path.read_text())
    if corruption == "missing_fold":
        path.unlink()
    else:
        if corruption == "mixed_variant":
            report["variant"] = "A3"
        elif corruption == "smoke":
            report["smoke"] = True
        else:
            from vestibular_fusion.evaluation.io import read_csv
            rows = read_csv(folder / "predictions.csv")
            rows[0]["sample_id"] = "different"
            write_csv(folder / "predictions.csv", rows)
            report["predictions_sha256"] = sha256_file(folder / "predictions.csv")
        write_json(path, report)
    with pytest.raises((FileNotFoundError, RuntimeError)):
        summarize(tmp_path, protocol, {}, "official", ["state"], splits)
    assert not (tmp_path / "summary").exists()
