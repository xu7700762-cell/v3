import math

import numpy as np
import pytest
import torch

from vestibular_fusion.model.severity_dynamics import (
    CONFIGS,
    NormalizedFractionalDoGKAN,
    SeverityDynamicsHead,
    ordered_summary,
)
from vestibular_fusion.training.severity_dynamics import severity_diagnostics
from vestibular_fusion.training.token_data import TokenExample, sample_hash, uniform_windows
from vestibular_fusion.training.token_pilot import restored_head, train_job


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_ordered_summary_has_directional_slope():
    increasing = torch.arange(17.0).reshape(1, 17, 1)
    forward = ordered_summary(increasing)
    reverse = ordered_summary(increasing.flip(1))
    assert forward.shape == (1, 3)
    assert torch.equal(forward[:, :2], reverse[:, :2])
    assert forward[0, 2] > 0 and reverse[0, 2] == -forward[0, 2]


@pytest.mark.parametrize("config", CONFIGS)
def test_severity_head_shape_gradient_and_no_reference_api(config):
    head = SeverityDynamicsHead(config)
    tokens = torch.randn(2, 17, 80, 525)
    logits = head(tokens)
    assert logits.shape == (2,) and logits.dtype == torch.float32
    logits.sum().backward()
    assert all(parameter.grad is not None for parameter in head.parameters())
    with pytest.raises(TypeError):
        head(tokens, anchors=torch.randn(2, 4, 80, 525))
    assert SeverityDynamicsHead(config)(torch.randn(1, 13, 80, 525)).shape == (1,)
    with pytest.raises(ValueError):
        head(torch.randn(1, 10, 80, 525))


def test_initialization_pairing_and_mlp_parameter_match():
    heads = {name: SeverityDynamicsHead(name) for name in CONFIGS}
    initial = {name: head.initialization() for name, head in heads.items()}
    assert len({value["base_sha256"] for value in initial.values()}) == 1
    assert len({initial[name]["residual_output_sha256"] for name in CONFIGS if name != "base"}) == 1
    dog_mapper = sum(p.numel() for p in heads["dog"].mapping.parameters())
    mlp_mapper = sum(p.numel() for p in heads["mlp"].mapping.parameters())
    assert abs(dog_mapper - mlp_mapper) / dog_mapper < 0.01
    assert heads["dog"].residual_gate_logit.sigmoid().item() == pytest.approx(0.25)


def test_dog_bases_are_independent_equal_scale_and_active():
    layer = NormalizedFractionalDoGKAN(525, 64, include_dog=True)
    values = torch.randn(4, 525)
    bases = layer.basis_features(values)
    assert len(bases) == 3
    for basis in bases:
        assert float(basis.detach().square().mean(-1).sqrt().mean()) == pytest.approx(1.0, abs=2e-4)
    assert torch.allclose(layer.fractional_orders(), torch.ones(15))
    assert torch.allclose(layer.basis_gate_logits.softmax(0) * 3, torch.ones(3))
    outputs = layer.component_outputs(values)
    assert all(float(value.detach().square().sum()) > 0 for value in outputs)
    layer(values).sum().backward()
    assert layer.basis_weight.grad[:, 2 * 525:].abs().sum() > 0
    assert layer.dog_scale_mix_logits.grad.abs().sum() > 0


def test_combined_logit_is_bounded_residual_formula():
    head = SeverityDynamicsHead("dog")
    parts = head.forward_components(torch.randn(2, 17, 80, 525))
    assert torch.allclose(parts["combined_logit"],
                          parts["base_logit"] + parts["residual_gate"] * parts["residual_logit"])
    assert 0 < parts["residual_gate"] < 1


def test_uniform_seventeen_requires_distinct_source_windows():
    assert uniform_windows(range(17), 17) == tuple(range(17))
    selected = uniform_windows(range(34), 17)
    assert len(selected) == len(set(selected)) == 17
    with pytest.raises(ValueError):
        uniform_windows(range(16), 17)


def test_generic_training_factory_saves_and_reloads(tmp_path):
    rng = np.random.default_rng(4)
    tokens = rng.normal(size=(40, 80, 525)).astype(np.float32)
    train = [TokenExample("train", f"segment_{i}", tuple(range(17)), i % 2, f"train/{i}")
             for i in range(4)]
    val = [TokenExample("val", f"segment_{i}", tuple(range(17, 34)), i % 2, f"val/{i}")
           for i in range(4)]
    protocol = {"seed": 2001, "batch_size": {"severity": 2},
                "microbatch_size": {"severity": 1}, "head_lr": 1e-3,
                "weight_decay": 1e-2, "grad_clip": 1.0, "max_epochs": 1,
                "patience_epochs": 10}
    identity = {"config": "base", "task": "severity", "dataset": "vrq",
                "protocol": protocol, "smoke": False,
                "train_sha256": sample_hash(train), "val_sha256": sample_hash(val)}
    factory = lambda name, seed: SeverityDynamicsHead(name, seed, 17)
    report = train_job(tmp_path, identity, tokens, train, val, torch.device("cpu"),
                       head_factory=factory, diagnostics_fn=severity_diagnostics)
    assert report["global_step"] == 2 and report["head_updated"]
    restored, _ = restored_head(tmp_path / "best.pt", identity, torch.device("cpu"), factory)
    assert isinstance(restored, SeverityDynamicsHead)
