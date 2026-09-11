import pytest
import torch

from vestibular_fusion.model.linear_probe import tensor_state_hash
from vestibular_fusion.model.vrq_boosted import VRQBoostedHead


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("task,shape", [("state", (3, 80, 525)),
                                         ("severity", (3, 11, 80, 525))])
def test_task_shapes_and_residual_addition(task, shape):
    base = VRQBoostedHead("base", task, 2001)
    head = VRQBoostedHead("dog", task, 2001, base.base_state())
    output = head.forward_outputs(torch.randn(*shape))
    assert output["binary_logit"].shape == (3,)
    assert torch.allclose(output["binary_logit"], output["base_logit"] + output["residual_logit"])


def test_residual_heads_share_and_freeze_exact_base():
    for task in ("state", "severity"):
        base = VRQBoostedHead("base", task, 2001)
        state = base.base_state()
        expected = tensor_state_hash(state)
        for config in ("mlp", "poly", "fractional", "dog"):
            head = VRQBoostedHead(config, task, 2001, state)
            assert tensor_state_hash(head.base_state()) == expected
            assert all(not parameter.requires_grad for name, parameter in head.named_parameters()
                       if name.startswith(head._base_prefixes()))
            assert all(parameter.requires_grad for name, parameter in head.named_parameters()
                       if not name.startswith(head._base_prefixes()))


def test_wrong_inputs_and_missing_base_are_rejected():
    with pytest.raises(ValueError):
        VRQBoostedHead("dog", "state", 2001)
    with pytest.raises(ValueError):
        VRQBoostedHead("base", "severity", 2001)(torch.randn(2, 10, 80, 525))
