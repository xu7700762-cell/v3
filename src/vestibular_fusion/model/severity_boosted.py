"""Two-stage severity model with one frozen shared base and residual readouts."""
from __future__ import annotations

import torch
from torch import nn

from .linear_probe import tensor_state_hash
from .main import ParameterMatchedMLP
from .severity_dynamics import (CONFIGS, NormalizedFractionalDoGKAN,
                                NormalizedPolynomialKAN, ordered_summary)


class BoostedSeverityHead(nn.Module):
    task = "severity"
    minimum_windows = 11
    smoke_window_count = 11

    def __init__(self, config: str, seed: int = 2001, base_state=None) -> None:
        super().__init__()
        if config not in CONFIGS:
            raise ValueError("Unknown boosted severity configuration")
        self.config = config
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 101)
            self.base_projection = nn.Sequential(nn.LayerNorm(525), nn.Linear(525, 64))
            torch.manual_seed(seed + 211)
            self.base_classifier = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 1))
            torch.manual_seed(seed + 223)
            self.base_regressor = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 1))
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
                torch.manual_seed(seed + 419)
                self.residual_regressor = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 1))
        if config != "base":
            if base_state is None:
                raise ValueError("Residual heads require a trained shared base state")
            self.load_and_freeze_base(base_state)
        elif base_state is not None:
            raise ValueError("Base training must start from its locked random initialization")

    @staticmethod
    def _base_prefixes():
        return ("base_projection.", "base_classifier.", "base_regressor.")

    def base_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items()
                if name.startswith(self._base_prefixes())}

    def load_and_freeze_base(self, state):
        expected = set(self.base_state())
        if set(state) != expected:
            raise RuntimeError("Shared base checkpoint tensors do not match")
        current = self.state_dict()
        current.update(state)
        self.load_state_dict(current, strict=True)
        for name, parameter in self.named_parameters():
            if name.startswith(self._base_prefixes()):
                parameter.requires_grad_(False)

    def _base_features(self, tokens):
        return ordered_summary(self.base_projection(tokens.mean(dim=2)))

    def _residual_features(self, tokens):
        return ordered_summary(self.mapping(tokens).mean(dim=2))

    def forward_outputs(self, tokens):
        if tokens.ndim != 4 or tokens.shape[1] < self.minimum_windows or tokens.shape[2:] != (80, 525):
            raise ValueError("Expected severity tokens [batch,at_least_11,80,525]")
        tokens = tokens.float()
        base = self._base_features(tokens)
        binary = self.base_classifier(base).squeeze(-1)
        score = self.base_regressor(base).squeeze(-1)
        result = {"base_logit": binary, "base_score": score}
        if self.config != "base":
            residual = self._residual_features(tokens)
            residual_logit = self.residual_classifier(residual).squeeze(-1)
            residual_score = self.residual_regressor(residual).squeeze(-1)
            result.update(residual_logit=residual_logit, residual_score=residual_score)
            binary, score = binary + residual_logit, score + residual_score
        result.update(binary_logit=binary, continuous_score=score)
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
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "shapes": {name: list(value.shape) for name, value in self.named_parameters()},
        }
