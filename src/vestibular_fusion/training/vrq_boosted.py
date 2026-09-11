"""Diagnostics for shared-base VRQ binary residual heads."""
from __future__ import annotations

import torch

from ..model.severity_dynamics import NormalizedFractionalDoGKAN
from .token_pilot import cpu_state, score_head, token_inputs


def boosted_diagnostics(head, tokens, examples, device, micro):
    with torch.no_grad():
        values = token_inputs(tokens, examples[:1], head.task, device)
        outputs = head.forward_outputs(values)
        result = {
            "task": head.task,
            "diagnostic_sample_windows": len(examples[0].indices),
            "base_logit_rms": float(outputs["base_logit"].square().mean().sqrt()),
            "base_parameters_trainable": any(
                parameter.requires_grad for name, parameter in head.named_parameters()
                if name.startswith(head._base_prefixes())
            ),
        }
        if head.config == "base":
            return result
        result["residual_logit_rms"] = float(outputs["residual_logit"].square().mean().sqrt())
        layer = head.mapping[1]
        if not isinstance(layer, NormalizedFractionalDoGKAN):
            return result
        mapped_input = head.mapping[0](values).reshape(-1, 525)[:256]
        bases = layer.basis_features(mapped_input)
        components = layer.component_outputs(mapped_input)
        result.update(
            fractional_order_mean=float(layer.fractional_orders().mean()),
            fractional_order_min=float(layer.fractional_orders().min()),
            fractional_order_max=float(layer.fractional_orders().max()),
            basis_gates=(layer.basis_gate_logits.softmax(0) * layer.basis_count).cpu().tolist(),
            basis_rms=[float(value.square().mean().sqrt()) for value in bases],
            component_output_rms=[float(value.square().mean().sqrt()) for value in components],
        )
        if layer.include_dog:
            result["dog_scale_mixture_mean"] = layer.dog_scale_mix_logits.softmax(-1).mean(0).cpu().tolist()
            state = cpu_state(head)
            layer.basis_weight[:, 2 * layer.input_dim:3 * layer.input_dim].zero_()
            _, result["counterfactual_no_dog"] = score_head(head, tokens, examples, device, micro)
            head.load_state_dict(state)
        return result
