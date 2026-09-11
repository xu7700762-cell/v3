import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from vestibular_fusion.model.token_probe import (CONFIGS, TokenReadout, FrozenTokenProbe,
                                                  statistics, window_summary, task_summary)
from vestibular_fusion.model.linear_probe import tensor_state_hash, build_probe
from vestibular_fusion.training.token_data import (TokenExample, uniform_eleven, source_split,
                                                   sample_hash, load_source_data)
from vestibular_fusion.training.token_pilot import (train_job, score_head, restored_head,
    verify_artifacts, token_cache, optimizer_step)
from vestibular_fusion.training.anchor_pilot import should_validate
from vestibular_fusion.evaluation.io import write_json, read_json
from vestibular_fusion.evaluation.token_pilot import summarize


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("task", ("state", "severity"))
def test_shapes_and_trainable_coverage(config, task):
    head = TokenReadout(config, task)
    tokens = torch.randn((2, 80, 525) if task == "state" else (2, 11, 80, 525))
    logits = head(tokens)
    assert logits.shape == (2,) and logits.dtype == torch.float32
    logits.sum().backward()
    for name, p in head.named_parameters():
        assert (p.grad is not None) == p.requires_grad, name
    width = 525 if config == "linear" else (480 if task == "state" else 1440) if config.startswith("R2") else 160
    assert head.classifier.in_features == width
    if "order1" in config:
        assert head.mapping[1].fractional_order_logit.item() == 0
    if "no_dog" in config:
        assert not head.mapping[1].dog_mix_logits.requires_grad


def test_hand_calculated_summaries_and_order():
    x = torch.tensor([[[1., 0.], [3., 2.], [5., 4.]]])
    result = window_summary(x)
    assert torch.allclose(result[:, :2], torch.tensor([[3., 2.]]))
    assert torch.allclose(result[:, 2:4], torch.full((1, 2), (8/3 + 1e-6)**.5))
    assert torch.equal(result[:, 4:], torch.tensor([[2., 2.]]))
    q = torch.arange(11.).reshape(1, 11, 1)
    a, b = task_summary(q), task_summary(q.flip(1))
    assert a[0, 0] == 5 and a[0, 2] == 8 and b[0, 2] == -8
    assert torch.equal(a[:, :2], b[:, :2])


def test_r0_r1_r2_pooling_order():
    tokens = torch.randn(2, 11, 80, 525)
    r0, r1, r2 = [TokenReadout(r + "_dog", "severity") for r in ("R0", "R1", "R2")]
    assert torch.equal(r0.features(tokens), r0.mapping(tokens.mean(-2).mean(1)))
    assert torch.equal(r1.features(tokens), r1.mapping(tokens).mean(-2).mean(1))
    assert torch.equal(r2.features(tokens), task_summary(window_summary(r2.mapping(tokens))))
    assert not torch.allclose(r0(tokens), r1(tokens))


def test_parameter_counts_and_pairing():
    a, b, c = [TokenReadout("R0_" + h, "state") for h in ("dog", "mlp", "poly")]
    assert a.initialization()["parameters"] == 170749
    assert b.initialization()["parameters"] == 170447
    assert a.initialization()["output_sha256"] == b.initialization()["output_sha256"] == c.initialization()["output_sha256"]
    for r in ("R1", "R2"):
        assert TokenReadout(r + "_dog", "state").initialization()["mapping_sha256"] == a.initialization()["mapping_sha256"]
    base = TokenReadout("R2_dog", "severity").initialization()
    for config in CONFIGS[-3:]:
        x = TokenReadout(config, "severity").initialization()
        assert x["head_sha256"] == base["head_sha256"]
        assert x["trainable_parameters"] < base["trainable_parameters"]


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.register_buffer("offset", torch.zeros(1))

    def forward_tokens(self, x):
        return (x.mean((1, 2)) * self.weight + self.offset)[:, None, None].expand(-1, 80, 525)


def test_frozen_parent_train_and_no_reference_api():
    model = FrozenTokenProbe(FakeEncoder(), TokenReadout("R2_dog", "state")).train()
    before = tensor_state_hash(model.encoder.state_dict())
    y = model(torch.randn(2, 30, 1280))
    y.sum().backward()
    assert not model.encoder.training
    assert all(p.grad is None for p in model.encoder.parameters())
    assert tensor_state_hash(model.encoder.state_dict()) == before
    with pytest.raises(TypeError):
        model(torch.randn(2, 30, 1280), anchors=torch.randn(4, 30, 1280))


def tiny_problem():
    rng = np.random.default_rng(8)
    tokens = rng.normal(size=(8, 80, 525)).astype(np.float32)
    train = [TokenExample("train", "segment_00", (i,), i % 2, f"train/{i}") for i in range(4)]
    val = [TokenExample("val", "segment_00", (i,), i % 2, f"val/{i}") for i in range(4, 8)]
    protocol = {"seed": 2001, "fold": "fold_1", "outer_test_scored": False,
                "batch_size": {"state": 2}, "microbatch_size": {"state": 1},
                "head_lr": .001, "weight_decay": .01, "grad_clip": 1., "max_epochs": 3, "patience_epochs": 10}
    identity = {"config": "R2_dog", "task": "state", "dataset": "vrq", "protocol": protocol,
                "smoke": False, "train_sha256": sample_hash(train), "val_sha256": sample_hash(val)}
    return tokens, train, val, identity


def test_early_best_and_resume_rollback_exact(tmp_path):
    tokens, train, val, ident = tiny_problem()
    device = torch.device("cpu")
    full = train_job(tmp_path / "full", ident, tokens, train, val, device)
    def interrupt(step):
        if step == 3:
            raise InterruptedError("deliberate partial epoch interruption")
    with pytest.raises(InterruptedError):
        train_job(tmp_path / "resumed", ident, tokens, train, val, device, step_callback=interrupt)
    resumed = train_job(tmp_path / "resumed", ident, tokens, train, val, device, resume=True)
    assert resumed["resume_events"][0]["restored_step"] == 2
    assert resumed["best_validation"] == full["best_validation"]
    a, _ = restored_head(tmp_path / "full/best.pt", ident, device)
    b, _ = restored_head(tmp_path / "resumed/best.pt", ident, device)
    assert tensor_state_hash(a.state_dict()) == tensor_state_hash(b.state_dict())
    checks = read_json(tmp_path / "resumed/history.json")["checks"]
    assert [r["step"] for r in checks] == list(range(1, 7))
    assert "validation" in checks[0]
    assert full["best_step"] == min(checks, key=lambda x: (x["validation"]["loss"], x["step"]))["step"]
    with pytest.raises(FileExistsError):
        train_job(tmp_path / "full", ident, tokens, train, val, device)
    with pytest.raises(RuntimeError):
        restored_head(tmp_path / "full/best.pt", {**ident, "config": "R2_mlp"}, device)
    with (tmp_path / "full/predictions.csv").open("a") as f:
        f.write("corruption")
    with pytest.raises(RuntimeError):
        verify_artifacts(tmp_path / "full", ident)


def test_label_and_other_sample_independence():
    tokens, _, val, _ = tiny_problem()
    head = TokenReadout("R2_mlp", "state")
    rows, _ = score_head(head, tokens, val, torch.device("cpu"), 1)
    new = [replace(e, label=1-e.label, subject_id="renamed") for e in val]
    again, _ = score_head(head, tokens, new, torch.device("cpu"), 1)
    assert [r["logit"] for r in rows] == [r["logit"] for r in again]
    changed = tokens.copy()
    changed[:4] *= 100
    repeat, _ = score_head(head, changed, val, torch.device("cpu"), 1)
    assert repeat == rows


def test_microbatch_loss_normalization():
    tokens, train, _, ident = tiny_problem()
    a, b = TokenReadout("R0_mlp", "state"), TokenReadout("R0_mlp", "state")
    p = ident["protocol"]
    opt_a = torch.optim.SGD(a.parameters(), lr=.01)
    opt_b = torch.optim.SGD(b.parameters(), lr=.01)
    optimizer_step(a, opt_a, tokens, train[:3], torch.device("cpu"), p, 2.)
    full = copy.deepcopy(p)
    full["microbatch_size"]["state"] = 3
    optimizer_step(b, opt_b, tokens, train[:3], torch.device("cpu"), full, 2.)
    for x, y in zip(a.parameters(), b.parameters()):
        assert torch.allclose(x, y, atol=1e-7, rtol=1e-5)


def test_source_loader_no_reference_metadata_or_outer_eeg(tmp_path, monkeypatch):
    from vestibular_fusion.training import token_data as data
    manifest = {"folds": {"fold_1": {"train_subjects": ["s1", "s2"], "val_subjects": ["s3"], "test_subjects": ["s4"]}},
                "subject_protocols": [{"subject_id": s, "final_task": "task01", "post_rest": None} for s in ("s1", "s2", "s3", "s4")],
                "audit": {"subjects": {s: {"ssq_label": i % 2} for i, s in enumerate(("s1", "s2", "s3"))}},
                "run_fingerprint_payload": {"mat_key": "data256", "inputs": {"mat_sha256": {
                    f"{s}_{session}.mat": "fake" for s in ("s1", "s2", "s3") for session in ("rest01", "rest02", "task01")}}}}
    write_json(tmp_path / "vrq/audit_manifest.json", manifest)
    write_json(tmp_path / "manifest.json", {"protocol_files": {"protocols/vrq/audit_manifest.json": {"sha256": "fake"}}})
    monkeypatch.setattr(data, "checked_file", lambda p, sha: {"file": p.name, "sha256": sha})
    seen = []
    def loader(path, key):
        seen.append(path.name)
        return np.random.default_rng(9).normal(size=(11, 30, 1280)).astype(np.float32)
    monkeypatch.setattr(data, "load_windows", loader)
    config = {"protocol_root": tmp_path, "paths": {"vrq_data_root": tmp_path}}
    windows, partitions, audit = load_source_data(config, "vrq")
    assert len(windows) == 99 and len(partitions["state"]["train"]) == 66
    assert all("s4" not in p for p in seen)
    assert all(e.session.startswith("segment_") for e in partitions["state"]["train"])
    assert audit["offline_transductive_subject_EA"] and not audit["reference_calibration"]
    manifest["reference_session"] = "nonexistent"
    for value in manifest["audit"]["subjects"].values():
        value["anchor_session"] = "nonsense"
        value["ssq_label"] = 1 - value["ssq_label"]
    write_json(tmp_path / "vrq/audit_manifest.json", manifest)
    changed, _, _ = load_source_data(config, "vrq")
    assert np.array_equal(windows, changed)


def test_cache_integrity(tmp_path):
    windows = np.random.default_rng(5).normal(size=(16, 30, 1280)).astype(np.float32)
    encoder = FakeEncoder().requires_grad_(False).eval()
    info = {"dataset": "test"}
    tokens, manifest = token_cache(tmp_path, windows, info, encoder, {}, torch.device("cpu"))
    assert tokens.shape == (16, 80, 525) and manifest["online_verified"]
    path = next((tmp_path / "cache").rglob("tokens.npy"))
    del tokens
    with path.open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(RuntimeError):
        token_cache(tmp_path, windows, info, encoder, {}, torch.device("cpu"))


def test_incomplete_summary_and_bad_inputs(tmp_path):
    assert summarize(tmp_path)["status"] == "partial"
    assert summarize(tmp_path, smoke=True)["full_matrix_completed"] is False
    assert len(CONFIGS) == 13
    with pytest.raises(ValueError):
        uniform_eleven(range(10))
    assert uniform_eleven(range(11)) == tuple(range(11))
    with pytest.raises(ValueError):
        TokenReadout("R2_dog", "severity")(torch.randn(1, 10, 80, 525))
    with pytest.raises(ValueError):
        source_split({"folds": {"fold_1": {"train_subjects": ["a"], "val_subjects": ["a"], "test_subjects": ["b"]}}}, "vrq")
    checkpoint = tmp_path / "bad.pt"
    checkpoint.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="SHA-256"):
        build_probe("A3", 2001, torch.device("cpu"), checkpoint_path=checkpoint,
                    expected_sha256="wrong", encoder_factory=FakeEncoder)


def test_first_update_best_is_not_discarded_on_ties(tmp_path, monkeypatch):
    from vestibular_fusion.training import token_pilot as runner
    tokens, train, val, identity = tiny_problem()
    frozen_rows, frozen_metrics = score_head(TokenReadout("R2_dog", "state"), tokens, val, torch.device("cpu"), 1)
    monkeypatch.setattr(runner, "score_head", lambda *args: (copy.deepcopy(frozen_rows), copy.deepcopy(frozen_metrics)))
    report = train_job(tmp_path / "early", identity, tokens, train, val, torch.device("cpu"))
    assert report["best_step"] == 1 and report["global_step"] == 6


def test_ea_is_label_free_but_transductive():
    from vestibular_fusion.data.features import fit_and_apply_subject_ea
    x = np.random.default_rng(30).normal(size=(3, 30, 1280)).astype(np.float32)
    aligned, _, _ = fit_and_apply_subject_ea(x)
    changed = x.copy()
    changed[1:, 0] *= 8
    other, _, _ = fit_and_apply_subject_ea(changed)
    assert not np.allclose(aligned[0], other[0])


def test_summary_rejects_foreign_protocol(tmp_path, monkeypatch):
    from vestibular_fusion.evaluation import token_pilot as evaluation
    folder = tmp_path / "R0_dog/state/vrq/fold_1"
    write_json(folder / "report.json", {})
    monkeypatch.setattr(evaluation, "verify_artifacts", lambda folder: {"identity": {"protocol": {}, "schema": "wrong"}})
    with pytest.raises(RuntimeError, match="locked pilot protocol"):
        summarize(tmp_path)
