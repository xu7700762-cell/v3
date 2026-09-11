from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolynomialKANLayer(nn.Module):
    """Degree-limited polynomial KAN projection used by the shared task head."""

    def __init__(self, input_dim: int, output_dim: int, degree: int = 2) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.degree = int(degree)
        if min(self.input_dim, self.output_dim, self.degree) <= 0:
            raise ValueError("KAN dimensions and degree must be positive")
        scale = 1.0 / math.sqrt(self.input_dim * self.degree)
        self.poly_weight = nn.Parameter(
            torch.randn(self.output_dim, self.input_dim * self.degree) * scale
        )
        self.bias = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        limited = torch.tanh(value)
        terms = [limited]
        for _ in range(1, self.degree):
            terms.append(terms[-1] * limited)
        return F.linear(
            torch.cat(terms, dim=-1),
            self.poly_weight.to(dtype=value.dtype),
            self.bias.to(dtype=value.dtype),
        )


class FractionalDoGPolynomialKANLayer(nn.Module):
    """Degree-2 polynomial KAN with fractional and feature-wise DoG bases."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        degree: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.degree = int(degree)
        if min(self.input_dim, self.output_dim) <= 0:
            raise ValueError("Residual polynomial KAN dimensions must be positive")
        if self.degree != 2:
            raise ValueError("Fractional DoG polynomial KAN requires degree=2")
        scale = 1.0 / math.sqrt(self.input_dim * 2)
        raw_weight = torch.randn(self.output_dim, self.input_dim * 2) * scale
        raw_quadratic = raw_weight[:, self.input_dim :].clone()
        raw_weight[:, self.input_dim :].mul_(0.5)
        self.poly_weight = nn.Parameter(raw_weight)
        self.bias = nn.Parameter(0.5 * raw_quadratic.sum(dim=1))
        self.log_input_scale = nn.Parameter(torch.zeros(self.input_dim))
        self.fractional_order_logit = nn.Parameter(torch.tensor(0.0))
        self.rational_logits = nn.Parameter(torch.zeros(self.degree))
        self.order_logits = nn.Parameter(torch.zeros(self.degree))
        self.linear_residual_logit = nn.Parameter(torch.tensor(0.0))
        self.dog_scale_logit = nn.Parameter(torch.tensor(0.0))
        self.dog_shift = nn.Parameter(torch.tensor(0.0))
        self.dog_mix_logits = nn.Parameter(torch.zeros(self.input_dim))

    def basis_features(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = torch.exp(0.5 * torch.tanh(self.log_input_scale))
        limited = torch.tanh(value * scale)
        fractional_order = 1.0 + 0.25 * torch.tanh(
            self.fractional_order_logit
        )
        epsilon = 1e-4
        fractional = torch.sign(limited) * (
            (limited.abs() + epsilon).pow(fractional_order)
            - epsilon**fractional_order
        )
        linear_mix = 0.1 * torch.tanh(self.linear_residual_logit)
        rational = 0.1 * torch.tanh(self.rational_logits)
        order_gain = 1.0 + 0.25 * torch.tanh(self.order_logits)
        first = (1.0 - linear_mix) * fractional + linear_mix * value
        second = 2.0 * fractional.square() - 1.0
        dog = self.dog_features(fractional)
        second = second + 0.25 * torch.tanh(self.dog_mix_logits) * dog
        denominator_base = fractional.square()
        terms = (first, second)
        stabilized = tuple(
            term * order_gain[index]
            / (1.0 + rational[index] * denominator_base)
            for index, term in enumerate(terms)
        )
        return (*stabilized, fractional, dog)

    def dog_features(self, fractional: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(0.5 * torch.tanh(self.dog_scale_logit))
        coordinate = (fractional - self.dog_shift) / scale
        return -coordinate * torch.exp(-0.5 * coordinate.square())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first, second, _, _ = self.basis_features(value.float())
        return F.linear(
            torch.cat([first, second], dim=-1).to(dtype=value.dtype),
            self.poly_weight.to(dtype=value.dtype),
            self.bias.to(dtype=value.dtype),
        )
