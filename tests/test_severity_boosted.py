import numpy as np
import pytest
import torch

from vestibular_fusion.model.linear_probe import tensor_state_hash
from vestibular_fusion.model.severity_boosted import BoostedSeverityHead
from vestibular_fusion.training.severity_boosted import make_optimizer_step, make_score_fn
from vestibular_fusion.training.severity_boosted_data import ContinuousSeverityExample


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def examples():
    return [ContinuousSeverityExample("s", f"segment_{index}", tuple(range(11)), index % 2,
                                      f"s/{index}", float(10 + index * 10))
            for index in range(4)]


def test_residual_loads_and_freezes_exact_shared_base():
    base = BoostedSeverityHead("base", 2001)
    state = base.base_state()
    expected = tensor_state_hash(state)
    for config in ("mlp", "poly", "fractional", "dog"):
        head = BoostedSeverityHead(config, 2001, state)
        assert tensor_state_hash(head.base_state()) == expected
        assert all(not parameter.requires_grad for name, parameter in head.named_parameters()
                   if name.startswith(head._base_prefixes()))
        assert all(parameter.requires_grad for name, parameter in head.named_parameters()
                   if not name.startswith(head._base_prefixes()))
    with pytest.raises(ValueError):
        BoostedSeverityHead("dog", 2001)


def test_binary_and_continuous_outputs_add_residuals():
    base = BoostedSeverityHead("base", 2001)
    head = BoostedSeverityHead("dog", 2001, base.base_state())
    output = head.forward_outputs(torch.randn(2, 13, 80, 525))
    assert torch.allclose(output["binary_logit"], output["base_logit"] + output["residual_logit"])
    assert torch.allclose(output["continuous_score"], output["base_score"] + output["residual_score"])


def test_continuous_loss_updates_residual_without_changing_base():
    tokens = np.random.default_rng(7).normal(size=(11, 80, 525)).astype(np.float32)
    base = BoostedSeverityHead("base", 2001)
    head = BoostedSeverityHead("dog", 2001, base.base_state())
    before = tensor_state_hash(head.base_state())
    optimizer = torch.optim.SGD([parameter for parameter in head.parameters() if parameter.requires_grad], lr=1e-3)
    protocol = {"loss": {"binary_BCE": 1.0, "continuous_Huber": 0.2,
                          "pairwise_rank_softplus": 0.1}, "grad_clip": 1.0}
    result = make_optimizer_step(25.0, 10.0)(
        head, optimizer, tokens, examples(), torch.device("cpu"), protocol, 1.0
    )
    assert result["continuous_Huber"] > 0 and result["pairwise_rank"] > 0
    assert tensor_state_hash(head.base_state()) == before
    assert all(parameter.grad is None for name, parameter in head.named_parameters()
               if name.startswith(head._base_prefixes()))
    assert all(parameter.grad is not None for name, parameter in head.named_parameters()
               if not name.startswith(head._base_prefixes()))


def test_continuous_scoring_does_not_pass_truth_to_model():
    tokens = np.random.default_rng(11).normal(size=(11, 80, 525)).astype(np.float32)
    head = BoostedSeverityHead("base", 2001)
    rows, result = make_score_fn(25.0, 10.0)(head, tokens, examples(), torch.device("cpu"), 1)
    assert len(rows) == 4 and result["metrics"]["n_subjects"] == 1
    assert all("predicted_path_score" in row and "path_score" in row for row in rows)
