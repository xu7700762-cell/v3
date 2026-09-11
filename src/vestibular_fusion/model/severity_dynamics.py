"""Reference-free path-level severity heads for the focused DoG experiment."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .linear_probe import tensor_state_hash
from .main import ParameterMatchedMLP


CONFIGS = ("base", "mlp", "poly", "fractional", "dog")


def _rms_normalize(value: torch.Tensor) -> torch.Tensor:
    return value / (value.square().mean(dim=-1, keepdim=True) + 1e-6).sqrt()


def ordered_summary(value: torch.Tensor) -> torch.Tensor:
    """Mean, population standard deviation and least-squares temporal slope."""
    if value.ndim != 3 or value.shape[1] < 2:
        raise ValueError("Expected ordered path features [batch,windows,features]")
    mean = value.mean(dim=1)
    std = ((value - mean[:, None]).square().mean(dim=1) + 1e-6).sqrt()
    time = torch.linspace(-1.0, 1.0, value.shape[1], dtype=value.dtype, device=value.device)
    slope = (value * time[None, :, None]).sum(dim=1) / time.square().sum()
    return torch.cat((mean, std, slope), dim=-1)


class NormalizedPolynomialKAN(nn.Module):
    """Degree-two KAN whose two bases use the same scale treatment as DoG."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim, self.output_dim = int(input_dim), int(output_dim)
        scale = 1.0 / math.sqrt(self.input_dim * 2)
        self.basis_weight = nn.Parameter(torch.randn(self.output_dim, self.input_dim * 2) * scale)
        self.bias = nn.Parameter(torch.zeros(self.output_dim))

    def basis_features(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        limited = torch.tanh(value)
        return _rms_normalize(limited), _rms_normalize(2.0 * limited.square() - 1.0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        bases = self.basis_features(value.float())
        return F.linear(torch.cat(bases, dim=-1).to(value.dtype), self.basis_weight.to(value.dtype),
                        self.bias.to(value.dtype))


class NormalizedFractionalDoGKAN(nn.Module):
    """Independent, normalized fractional-polynomial and multi-scale DoG bases."""

    def __init__(self, input_dim: int, output_dim: int, *, include_dog: bool,
                 groups: int = 15) -> None:
        super().__init__()
        self.input_dim, self.output_dim = int(input_dim), int(output_dim)
        self.include_dog, self.groups = bool(include_dog), int(groups)
        if self.input_dim % self.groups:
            raise ValueError("input_dim must be divisible by groups")
        self.basis_count = 3 if self.include_dog else 2
        scale = 1.0 / math.sqrt(self.input_dim * self.basis_count)
        self.basis_weight = nn.Parameter(
            torch.randn(self.output_dim, self.input_dim * self.basis_count) * scale
        )
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        self.log_input_scale = nn.Parameter(torch.zeros(self.input_dim))
        initial_order_logit = math.log(0.375 / 0.625)  # q=1 inside [0.7, 1.5]
        self.fractional_order_logits = nn.Parameter(torch.full((self.groups,), initial_order_logit))
        self.basis_gate_logits = nn.Parameter(torch.zeros(self.basis_count))
        if self.include_dog:
            self.dog_shift_logits = nn.Parameter(torch.zeros(self.groups))
            self.dog_scale_mix_logits = nn.Parameter(torch.zeros(self.groups, 3))
            self.register_buffer("dog_scales", torch.tensor((0.5, 1.0, 2.0)))

    def _expanded(self, grouped: torch.Tensor) -> torch.Tensor:
        return grouped.repeat_interleave(self.input_dim // self.groups)

    def fractional_orders(self) -> torch.Tensor:
        return 0.7 + 0.8 * self.fractional_order_logits.sigmoid()

    def basis_features(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        scale = torch.exp(0.5 * torch.tanh(self.log_input_scale))
        limited = torch.tanh(value * scale)
        order = self._expanded(self.fractional_orders())
        epsilon = 1e-4
        fractional = torch.sign(limited) * (
            (limited.abs() + epsilon).pow(order) - epsilon**order
        )
        bases = [fractional, 2.0 * fractional.square() - 1.0]
        if self.include_dog:
            shift = 0.5 * self._expanded(self.dog_shift_logits.tanh())
            mix = self.dog_scale_mix_logits.softmax(dim=-1)
            mix = mix.repeat_interleave(self.input_dim // self.groups, dim=0)
            coordinate = (fractional[..., None] - shift[..., None]) / self.dog_scales
            bank = -coordinate * torch.exp(-0.5 * coordinate.square())
            bases.append((bank * mix).sum(dim=-1))
        gates = self.basis_gate_logits.softmax(dim=0) * self.basis_count
        return tuple(_rms_normalize(basis) * gates[index] for index, basis in enumerate(bases))

    def component_outputs(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        bases = self.basis_features(value.float())
        chunks = self.basis_weight.split(self.input_dim, dim=1)
        return tuple(F.linear(basis, weight) for basis, weight in zip(bases, chunks))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        bases = self.basis_features(value.float())
        return F.linear(torch.cat(bases, dim=-1).to(value.dtype), self.basis_weight.to(value.dtype),
                        self.bias.to(value.dtype))


class SeverityDynamicsHead(nn.Module):
    """Stable FEMBA path baseline with a matched nonlinear temporal residual."""

    task = "severity"

    def __init__(self, config: str, seed: int = 2001, minimum_windows: int = 11) -> None:
        super().__init__()
        if config not in CONFIGS:
            raise ValueError("Unknown severity dynamics configuration")
        self.config, self.minimum_windows = config, int(minimum_windows)
        self.smoke_window_count = self.minimum_windows
        if self.minimum_windows < 2:
            raise ValueError("Severity dynamics requires at least two windows")
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
                self.residual_gate_logit = nn.Parameter(torch.tensor(math.log(0.25 / 0.75)))

    def _components(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if tokens.ndim != 4 or tokens.shape[1] < self.minimum_windows or tokens.shape[2:] != (80, 525):
            raise ValueError(
                f"Expected severity tokens [batch,at_least_{self.minimum_windows},80,525]"
            )
        tokens = tokens.float()
        base_windows = self.base_projection(tokens.mean(dim=2))
        base_logit = self.base_classifier(ordered_summary(base_windows)).squeeze(-1)
        if self.config == "base":
            return base_logit, None
        residual_windows = self.mapping(tokens).mean(dim=2)
        residual_logit = self.residual_classifier(ordered_summary(residual_windows)).squeeze(-1)
        return base_logit, residual_logit

    def forward_components(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        base, residual = self._components(tokens)
        result = {"base_logit": base}
        if residual is not None:
            gate = self.residual_gate_logit.sigmoid()
            result.update(residual_logit=residual, residual_gate=gate,
                          combined_logit=base + gate * residual)
        else:
            result["combined_logit"] = base
        return result

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            return self.forward_components(tokens)["combined_logit"]

    def initialization(self) -> dict:
        return {
            "head_sha256": tensor_state_hash(self.state_dict()),
            "base_sha256": tensor_state_hash({k: v for k, v in self.state_dict().items()
                                                if k.startswith("base_")}),
            "mapping_sha256": tensor_state_hash(self.mapping.state_dict())
            if hasattr(self, "mapping") else None,
            "residual_output_sha256": tensor_state_hash(self.residual_classifier.state_dict())
            if hasattr(self, "residual_classifier") else None,
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "shapes": {name: list(p.shape) for name, p in self.named_parameters()},
        }
