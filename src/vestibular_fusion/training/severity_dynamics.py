"""Diagnostics for the reference-free severity dynamics heads."""
import math

import torch

from ..model.severity_dynamics import NormalizedFractionalDoGKAN
from .token_pilot import cpu_state, score_head, token_inputs


@torch.no_grad()
def severity_diagnostics(head, tokens, examples, device, micro):
    result = {"minimum_windows": head.minimum_windows,
              "diagnostic_sample_windows": len(examples[0].indices)}
    values = token_inputs(tokens, examples[:1], head.task, device)
    components = head.forward_components(values)
    result["base_logit_rms"] = float(components["base_logit"].square().mean().sqrt())
    if "residual_logit" in components:
        result.update(
            residual_gate=float(components["residual_gate"]),
            residual_logit_rms=float(components["residual_logit"].square().mean().sqrt()),
        )
    if not hasattr(head, "mapping") or not isinstance(head.mapping[1], NormalizedFractionalDoGKAN):
        return result
    layer = head.mapping[1]
    mapped_input = head.mapping[0](values).reshape(-1, 525)[:256]
    bases = layer.basis_features(mapped_input)
    outputs = layer.component_outputs(mapped_input)
    result.update(
        fractional_order_mean=float(layer.fractional_orders().mean()),
        fractional_order_min=float(layer.fractional_orders().min()),
        fractional_order_max=float(layer.fractional_orders().max()),
        basis_gates=(layer.basis_gate_logits.softmax(0) * layer.basis_count).cpu().tolist(),
        basis_rms=[float(value.square().mean().sqrt()) for value in bases],
        component_output_rms=[float(value.square().mean().sqrt()) for value in outputs],
        diagnostic_tokens=len(mapped_input),
    )
    if not layer.include_dog:
        return result
    result["dog_scale_mixture_mean"] = layer.dog_scale_mix_logits.softmax(-1).mean(0).cpu().tolist()
    state = cpu_state(head)
    dog_slice = slice(2 * layer.input_dim, 3 * layer.input_dim)
    q_one = math.log(0.375 / 0.625)
    for mode in ("no_dog", "order1", "order1_no_dog"):
        head.load_state_dict(state)
        if "no_dog" in mode:
            layer.basis_weight[:, dog_slice].zero_()
        if "order1" in mode:
            layer.fractional_order_logits.fill_(q_one)
        _, result["counterfactual_" + mode] = score_head(head, tokens, examples, device, micro)
    head.load_state_dict(state)
    return result
