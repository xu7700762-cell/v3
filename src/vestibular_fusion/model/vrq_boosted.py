"""Shared-base residual heads for VRQ state and binary severity development."""
from __future__ import annotations

import torch
from torch import nn

from .linear_probe import tensor_state_hash
from .main import ParameterMatchedMLP
from .severity_dynamics import (CONFIGS, NormalizedFractionalDoGKAN,
                                NormalizedPolynomialKAN, ordered_summary)
from .token_probe import window_summary


class VRQBoostedHead(nn.Module):
    """Frozen selected base plus a trainable, task-matched nonlinear residual."""

    smoke_window_count = 11

    def __init__(self, config: str, task: str, seed: int = 2001, base_state=None) -> None:
        super().__init__()
        if config not in CONFIGS or task not in ("state", "severity"):
            raise ValueError("Unknown VRQ boosted configuration")
        self.config, self.task = config, task
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 101)
            self.base_projection = nn.Sequential(nn.LayerNorm(525), nn.Linear(525, 64))
            torch.manual_seed(seed + 211)
            self.base_classifier = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 1))
            if config != "base":
                torch.manual_seed(seed + 307)
                mapper = (
                    ParameterMatchedMLP(525, 64, 172) if config == "mlp" else
                    NormalizedPolynomialKAN(525, 64) if config == "poly" else
                    NormalizedFractionalDoGKAN(525, 64, include_dog=config == "dog")
                )
                self.mapping = nn.Sequential(nn.LayerNorm(525), mapper, nn.LayerNorm(64))
                torch.manual_seed(seed + 401)
                self.residual_classifier = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 1))
        if config == "base":
            if base_state is not None:
                raise ValueError("Base training must start from its locked initialization")
        else:
            if base_state is None:
                raise ValueError("Residual heads require a trained shared base state")
            self.load_and_freeze_base(base_state)

    @staticmethod
    def _base_prefixes():
        return ("base_projection.", "base_classifier.")

    def base_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items()
                if name.startswith(self._base_prefixes())}

    def load_and_freeze_base(self, state):
        if set(state) != set(self.base_state()):
            raise RuntimeError("Shared base checkpoint tensors do not match")
        current = self.state_dict()
        current.update(state)
        self.load_state_dict(current, strict=True)
        for name, parameter in self.named_parameters():
            if name.startswith(self._base_prefixes()):
                parameter.requires_grad_(False)

    def _summarize(self, values):
        return window_summary(values) if self.task == "state" else ordered_summary(values)

    def _base_features(self, tokens):
        if self.task == "state":
            return self._summarize(self.base_projection(tokens))
        return self._summarize(self.base_projection(tokens.mean(dim=2)))

    def _residual_features(self, tokens):
        mapped = self.mapping(tokens)
        if self.task == "severity":
            mapped = mapped.mean(dim=2)
        return self._summarize(mapped)

    def forward_outputs(self, tokens):
        expected = 3 if self.task == "state" else 4
        if tokens.ndim != expected or tokens.shape[-2:] != (80, 525):
            raise ValueError("Expected state [B,80,525] or severity [B,11,80,525] tokens")
        if self.task == "severity" and tokens.shape[1] != 11:
            raise ValueError("VRQ severity requires exactly eleven ordered windows")
        tokens = tokens.float()
        base = self.base_classifier(self._base_features(tokens)).squeeze(-1)
        result = {"base_logit": base, "binary_logit": base}
        if self.config != "base":
            residual = self.residual_classifier(self._residual_features(tokens)).squeeze(-1)
            result.update(residual_logit=residual, binary_logit=base + residual)
        return result

    def forward(self, tokens):
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            return self.forward_outputs(tokens)["binary_logit"]

    def initialization(self):
        state = self.state_dict()
        return {
            "head_sha256": tensor_state_hash(state),
            "base_sha256": tensor_state_hash(self.base_state()),
            "mapping_sha256": tensor_state_hash(self.mapping.state_dict())
            if hasattr(self, "mapping") else None,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters()
                                        if parameter.requires_grad),
            "shapes": {name: list(value.shape) for name, value in self.named_parameters()},
        }
