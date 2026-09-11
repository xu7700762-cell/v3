from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .encoder import TemporalEncoder, load_pretrained_checkpoint
from .kan import FractionalDoGPolynomialKANLayer, PolynomialKANLayer


class ParameterMatchedMLP(nn.Sequential):
    """Two-layer MLP matched to the degree-2 KAN parameter count."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 245) -> None:
        super().__init__(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )


class PRISMResidualSeverityHead(nn.Module):
    """Stable FEMBA severity baseline with a bounded direct KAN residual."""

    def __init__(self) -> None:
        super().__init__()
        self.residual_bound = 0.1
        self.base_head = nn.Linear(3, 1, bias=False)
        self.correction_norm = nn.LayerNorm(160, elementwise_affine=False)
        self.correction_head = nn.Linear(160, 1, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("base_center", torch.zeros(3))
        self.register_buffer("base_scale", torch.ones(3))
        with torch.no_grad():
            self.base_head.weight.mul_(0.1)
            self.correction_head.weight.zero_()

    @staticmethod
    def base_feature(
        task_pooled: torch.Tensor, task_token_dynamics: torch.Tensor
    ) -> torch.Tensor:
        if task_pooled.ndim != 3 or task_pooled.shape[-1] != 525:
            raise ValueError(
                f"Expected FEMBA task features [batch,time,525], got {tuple(task_pooled.shape)}"
            )
        if (
            task_token_dynamics.ndim != 3
            or task_token_dynamics.shape[:2] != task_pooled.shape[:2]
            or task_token_dynamics.shape[-1] != 2
        ):
            raise ValueError(
                "Expected FEMBA token dynamics [batch,time,2] aligned with task features, "
                f"got {tuple(task_token_dynamics.shape)}"
            )
        centered = task_pooled - task_pooled.mean(dim=1, keepdim=True)
        pooled_dispersion = torch.sqrt(torch.mean(centered.square(), dim=(1, 2)))
        token_velocity = task_token_dynamics[..., 0].mean(dim=1)
        span = max(1, (task_token_dynamics.shape[1] + 1) // 3)
        drift_progression = (
            task_token_dynamics[:, -span:, 1].mean(dim=1)
            - task_token_dynamics[:, :span, 1].mean(dim=1)
        )
        return torch.stack(
            [pooled_dispersion, token_velocity, drift_progression], dim=-1
        )

    def normalized_base_feature(
        self, task_pooled: torch.Tensor, task_token_dynamics: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.base_feature(task_pooled, task_token_dynamics) - self.base_center
        ) / self.base_scale

    def set_base_normalization(
        self, center: torch.Tensor | list[float], scale: torch.Tensor | list[float]
    ) -> None:
        center_tensor = torch.as_tensor(center, dtype=self.base_center.dtype)
        scale_tensor = torch.as_tensor(scale, dtype=self.base_scale.dtype)
        if center_tensor.shape != (3,) or scale_tensor.shape != (3,):
            raise ValueError("Severity base normalization requires three features")
        if not torch.isfinite(center_tensor).all() or not torch.isfinite(scale_tensor).all():
            raise ValueError("Severity base normalization must be finite")
        if bool(torch.any(scale_tensor <= 0.0)):
            raise ValueError("Severity base normalization scale must be positive")
        self.base_center.copy_(center_tensor)
        self.base_scale.copy_(scale_tensor)

    def kan_features(
        self, calibrated: torch.Tensor, anchor_scale: torch.Tensor
    ) -> torch.Tensor:
        scale = anchor_scale.reshape(-1, 1).clamp_min(1e-6)
        return calibrated.abs().mean(dim=1) / scale

    def diagnostic_features(
        self,
        task_pooled: torch.Tensor,
        task_token_dynamics: torch.Tensor,
        calibrated: torch.Tensor,
        anchor_scale: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                self.normalized_base_feature(task_pooled, task_token_dynamics),
                self.kan_features(calibrated, anchor_scale),
            ],
            dim=-1,
        )

    def forward(
        self,
        task_pooled: torch.Tensor,
        task_token_dynamics: torch.Tensor,
        calibrated: torch.Tensor,
        anchor_scale: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base_head(
            self.normalized_base_feature(task_pooled, task_token_dynamics)
        )
        correction = self.residual_bound * torch.tanh(
                self.correction_head(
                    self.correction_norm(self.kan_features(calibrated, anchor_scale))
                )
        )
        return (base + correction + self.output_bias).squeeze(-1)


class FEMBAKANMultiTaskModel(nn.Module):
    """Frozen pretrained FEMBA with one shared projection and two task heads."""

    def __init__(
        self,
        encoder: nn.Module | None = None,
        *,
        latent_dim: int = 160,
        kan_degree: int = 2,
        head_dropout: float = 0.2,
        projection_variant: str = "fractional_dog_polykan",
        seed: int = 2001,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else TemporalEncoder()
        self.input_dim = 525
        self.latent_dim = int(latent_dim)
        self.kan_degree = int(kan_degree)
        self.head_dropout_rate = float(head_dropout)
        self.projection_variant = str(projection_variant)
        if self.projection_variant not in {
            "kan",
            "mlp",
            "fractional_dog_polykan",
        }:
            raise ValueError(
                "projection_variant must be 'kan', 'mlp', or "
                "'fractional_dog_polykan'"
            )
        with torch.random.fork_rng(devices=[]):
            self.input_norm = nn.LayerNorm(self.input_dim)
            torch.manual_seed(int(seed) + 101)
            if self.projection_variant == "kan":
                self.shared_kan = PolynomialKANLayer(
                    self.input_dim, self.latent_dim, degree=self.kan_degree
                )
            elif self.projection_variant == "mlp":
                self.shared_kan = ParameterMatchedMLP(
                    self.input_dim, self.latent_dim
                )
            else:
                self.shared_kan = FractionalDoGPolynomialKANLayer(
                    self.input_dim,
                    self.latent_dim,
                    degree=self.kan_degree,
                )
            self.latent_norm = nn.LayerNorm(self.latent_dim)
            self.head_dropout = nn.Dropout(self.head_dropout_rate)
            torch.manual_seed(int(seed) + 211)
            self.state_head = nn.Linear(self.latent_dim + 1, 1)
            torch.manual_seed(int(seed) + 307)
            self.severity_head = PRISMResidualSeverityHead()
            with torch.no_grad():
                self.state_head.weight[:, : self.latent_dim].mul_(0.1)
                self.state_head.weight[:, -1].fill_(1.0)
                self.state_head.bias.fill_(-1.4)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True) -> "FEMBAKANMultiTaskModel":
        super().train(mode)
        self.encoder.eval()
        return self

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        *,
        latent_dim: int = 160,
        kan_degree: int = 2,
        head_dropout: float = 0.2,
        projection_variant: str = "fractional_dog_polykan",
        seed: int = 2001,
    ) -> tuple["FEMBAKANMultiTaskModel", dict]:
        model = cls(
            latent_dim=latent_dim,
            kan_degree=kan_degree,
            head_dropout=head_dropout,
            projection_variant=projection_variant,
            seed=seed,
        )
        load_info = load_pretrained_checkpoint(model.encoder, Path(checkpoint_path))
        if int(load_info["loaded_keys"]) != 83 or any(
            load_info[name] for name in ("missing_keys", "unexpected_keys", "skipped_keys")
        ):
            raise RuntimeError(f"Incomplete pretrained FEMBA checkpoint: {load_info}")
        return model, load_info

    def pool_windows(self, windows: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.extract_frozen_features(windows)
        return pooled

    @staticmethod
    def token_dynamics(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-2:] != (80, 525):
            raise ValueError(
                f"Expected FEMBA tokens [batch,80,525], got {tuple(tokens.shape)}"
            )
        velocity = torch.sqrt(
            torch.mean((tokens[:, 1:] - tokens[:, :-1]).square(), dim=(1, 2))
        )
        third = tokens.shape[1] // 3
        drift = torch.sqrt(
            torch.mean(
                (
                    tokens[:, -third:].mean(dim=1)
                    - tokens[:, :third].mean(dim=1)
                ).square(),
                dim=1,
            )
        )
        return torch.stack([velocity, drift], dim=-1)

    def extract_frozen_features(
        self, windows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if windows.ndim != 3 or windows.shape[-2:] != (30, 1280):
            raise ValueError(f"Expected [batch,30,1280], got {tuple(windows.shape)}")
        tokens = self.encoder.forward_tokens(windows.float())
        if tokens.shape[-2:] != (80, self.input_dim):
            raise ValueError(f"Expected FEMBA tokens [batch,80,525], got {tuple(tokens.shape)}")
        return tokens.mean(dim=1), self.token_dynamics(tokens)

    def encode_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.ndim != 2 or pooled.shape[-1] != self.input_dim:
            raise ValueError(f"Expected pooled FEMBA features [batch,525], got {tuple(pooled.shape)}")
        return self.latent_norm(self.shared_kan(self.input_norm(pooled.float())))

    def encode_windows(self, windows: torch.Tensor) -> torch.Tensor:
        return self.encode_pooled(self.pool_windows(windows))

    def _encode_sets(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.shape[-2:] != (30, 1280):
            raise ValueError(f"Unexpected window shape: {tuple(windows.shape)}")
        if windows.ndim == 4:
            batch, count = windows.shape[:2]
            return self.encode_windows(windows.reshape(-1, 30, 1280)).reshape(
                batch, count, self.latent_dim
            )
        if windows.ndim == 5:
            batch, count, context = windows.shape[:3]
            return self.encode_windows(windows.reshape(-1, 30, 1280)).reshape(
                batch, count, context, self.latent_dim
            ).mean(dim=2)
        raise ValueError(
            f"Expected [batch,windows,30,1280] or [batch,windows,context,30,1280], "
            f"got {tuple(windows.shape)}"
        )

    def _pool_sets(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.shape[-2:] != (30, 1280):
            raise ValueError(f"Unexpected window shape: {tuple(windows.shape)}")
        if windows.ndim == 4:
            batch, count = windows.shape[:2]
            return self.pool_windows(windows.reshape(-1, 30, 1280)).reshape(
                batch, count, self.input_dim
            )
        if windows.ndim == 5:
            batch, count, context = windows.shape[:3]
            return self.pool_windows(windows.reshape(-1, 30, 1280)).reshape(
                batch, count, context, self.input_dim
            )
        raise ValueError(
            f"Expected [batch,windows,30,1280] or [batch,windows,context,30,1280], "
            f"got {tuple(windows.shape)}"
        )

    def _pool_and_dynamics_sets(
        self, windows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if windows.shape[-2:] != (30, 1280):
            raise ValueError(f"Unexpected window shape: {tuple(windows.shape)}")
        if windows.ndim == 4:
            batch, count = windows.shape[:2]
            pooled, dynamics = self.extract_frozen_features(
                windows.reshape(-1, 30, 1280)
            )
            return (
                pooled.reshape(batch, count, self.input_dim),
                dynamics.reshape(batch, count, 2),
            )
        if windows.ndim == 5:
            batch, count, context = windows.shape[:3]
            pooled, dynamics = self.extract_frozen_features(
                windows.reshape(-1, 30, 1280)
            )
            return (
                pooled.reshape(batch, count, context, self.input_dim),
                dynamics.reshape(batch, count, context, 2),
            )
        raise ValueError(
            f"Expected [batch,windows,30,1280] or [batch,windows,context,30,1280], "
            f"got {tuple(windows.shape)}"
        )

    def _encode_pooled_sets(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.shape[-1] != self.input_dim:
            raise ValueError(f"Unexpected pooled feature shape: {tuple(pooled.shape)}")
        if pooled.ndim == 3:
            batch, count = pooled.shape[:2]
            return self.encode_pooled(pooled.reshape(-1, self.input_dim)).reshape(
                batch, count, self.latent_dim
            )
        if pooled.ndim == 4:
            batch, count, context = pooled.shape[:3]
            return self.encode_pooled(pooled.reshape(-1, self.input_dim)).reshape(
                batch, count, context, self.latent_dim
            ).mean(dim=2)
        raise ValueError(
            f"Expected [batch,windows,525] or [batch,windows,context,525], got {tuple(pooled.shape)}"
        )

    def encode_contexts(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 4 or windows.shape[-2:] != (30, 1280):
            raise ValueError(f"Expected [batch,context,30,1280], got {tuple(windows.shape)}")
        batch, context = windows.shape[:2]
        return self.encode_windows(windows.reshape(-1, 30, 1280)).reshape(
            batch, context, self.latent_dim
        ).mean(dim=1)

    def encode_pooled_contexts(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.ndim != 3 or pooled.shape[-1] != self.input_dim:
            raise ValueError(f"Expected [batch,context,525], got {tuple(pooled.shape)}")
        batch, context = pooled.shape[:2]
        return self.encode_pooled(pooled.reshape(-1, self.input_dim)).reshape(
            batch, context, self.latent_dim
        ).mean(dim=1)

    @staticmethod
    def _statistics_from_anchor_embeddings(
        embeddings: torch.Tensor, single: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        center = embeddings.mean(dim=1)
        scale = torch.sqrt(
            torch.mean((embeddings - center.unsqueeze(1)).square(), dim=(1, 2))
        ).clamp_min(1e-6)
        return (center[0], scale[0]) if single else (center, scale)

    def anchor_statistics(self, anchor_windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if anchor_windows.ndim == 3:
            anchor_windows = anchor_windows.unsqueeze(0)
            single = True
        elif (
            anchor_windows.ndim == 4
            and anchor_windows.shape[0] == 4
            and anchor_windows.shape[1] == 3
        ):
            anchor_windows = anchor_windows.unsqueeze(0)
            single = True
        else:
            single = False
        if anchor_windows.ndim not in (4, 5) or anchor_windows.shape[1] != 4:
            raise ValueError(
                "Expected four anchors per subject with optional context, "
                f"got {tuple(anchor_windows.shape)}"
            )
        return self._statistics_from_anchor_embeddings(
            self._encode_sets(anchor_windows), single
        )

    def anchor_statistics_from_pooled(
        self, anchor_pooled: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if anchor_pooled.ndim == 2:
            anchor_pooled = anchor_pooled.unsqueeze(0)
            single = True
        elif (
            anchor_pooled.ndim == 3
            and anchor_pooled.shape[0] == 4
            and anchor_pooled.shape[1] == 3
        ):
            anchor_pooled = anchor_pooled.unsqueeze(0)
            single = True
        else:
            single = False
        if anchor_pooled.ndim not in (3, 4) or anchor_pooled.shape[1] != 4:
            raise ValueError(
                "Expected four pooled anchors per subject with optional context, "
                f"got {tuple(anchor_pooled.shape)}"
            )
        return self._statistics_from_anchor_embeddings(
            self._encode_pooled_sets(anchor_pooled), single
        )

    def anchor_center(self, anchor_windows: torch.Tensor) -> torch.Tensor:
        center, _ = self.anchor_statistics(anchor_windows)
        return center

    def calibrated_embeddings(
        self, windows: torch.Tensor, anchor_windows: torch.Tensor
    ) -> torch.Tensor:
        if windows.ndim == 3:
            values = self.encode_windows(windows)
            center = self.anchor_center(anchor_windows)
            if center.ndim != 1:
                raise ValueError("Single-subject windows require one four-anchor set")
            return values - center.unsqueeze(0)
        values = self._encode_sets(windows)
        center = self.anchor_center(anchor_windows)
        if center.ndim == 1:
            center = center.unsqueeze(0)
        if values.shape[0] != center.shape[0]:
            raise ValueError("Window and anchor subject batches are not aligned")
        return values - center.unsqueeze(1)

    def state_features(
        self,
        calibrated: torch.Tensor,
        anchor_scale: torch.Tensor,
        *,
        apply_dropout: bool,
    ) -> torch.Tensor:
        radius = torch.sqrt(torch.mean(calibrated.square(), dim=-1, keepdim=True))
        while anchor_scale.ndim < radius.ndim:
            anchor_scale = anchor_scale.unsqueeze(-1)
        scaled_radius = radius / anchor_scale
        magnitude = calibrated.abs()
        if apply_dropout:
            magnitude = self.head_dropout(magnitude)
        return torch.cat([magnitude, scaled_radius], dim=-1)

    def state_logits_from_calibrated(
        self,
        calibrated: torch.Tensor,
        anchor_scale: torch.Tensor,
        *,
        apply_dropout: bool = True,
    ) -> torch.Tensor:
        return self.state_head(
            self.state_features(calibrated, anchor_scale, apply_dropout=apply_dropout)
        ).squeeze(-1)

    def state_logits(self, windows: torch.Tensor, anchor_windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim == 3:
            values = self.encode_windows(windows)
            center, scale = self.anchor_statistics(anchor_windows)
            calibrated = values - center.unsqueeze(0)
        else:
            values = self._encode_sets(windows)
            center, scale = self.anchor_statistics(anchor_windows)
            calibrated = values - center.unsqueeze(1)
        return self.state_logits_from_calibrated(calibrated, scale)

    def state_logits_from_pooled(
        self, pooled: torch.Tensor, anchor_pooled: torch.Tensor
    ) -> torch.Tensor:
        logits, _ = self.state_logits_and_deviation_from_pooled(pooled, anchor_pooled)
        return logits

    def state_logits_and_deviation_from_pooled(
        self, pooled: torch.Tensor, anchor_pooled: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pooled.ndim == 2:
            values = self.encode_pooled(pooled)
            center, scale = self.anchor_statistics_from_pooled(anchor_pooled)
            calibrated = values - center.unsqueeze(0)
        else:
            values = self._encode_pooled_sets(pooled)
            center, scale = self.anchor_statistics_from_pooled(anchor_pooled)
            calibrated = values - center.unsqueeze(1)
        features = self.state_features(calibrated, scale, apply_dropout=True)
        return self.state_head(features).squeeze(-1), features[..., -1]

    def severity_features(
        self,
        task_pooled: torch.Tensor,
        task_token_dynamics: torch.Tensor,
        calibrated: torch.Tensor,
        anchor_scale: torch.Tensor,
    ) -> torch.Tensor:
        if calibrated.ndim != 3:
            raise ValueError(
                f"Expected calibrated task embeddings [batch,time,latent], got {tuple(calibrated.shape)}"
            )
        return self.severity_head.diagnostic_features(
            task_pooled, task_token_dynamics, calibrated, anchor_scale
        )

    def severity_logits(
        self,
        task_windows: torch.Tensor,
        anchor_windows: torch.Tensor,
        *,
        detach_shared: bool = False,
    ) -> torch.Tensor:
        task_pooled, task_token_dynamics = self._pool_and_dynamics_sets(task_windows)
        return self.severity_logits_from_pooled(
            task_pooled,
            self._pool_sets(anchor_windows),
            task_token_dynamics,
            detach_shared=detach_shared,
        )

    def severity_logits_from_pooled(
        self,
        task_pooled: torch.Tensor,
        anchor_pooled: torch.Tensor,
        task_token_dynamics: torch.Tensor,
        *,
        detach_shared: bool = False,
    ) -> torch.Tensor:
        task_embeddings = self._encode_pooled_sets(task_pooled)
        center, scale = self.anchor_statistics_from_pooled(anchor_pooled)
        calibrated = task_embeddings - center.unsqueeze(1)
        task_base = task_pooled.mean(dim=2) if task_pooled.ndim == 4 else task_pooled
        dynamics_base = (
            task_token_dynamics.mean(dim=2)
            if task_token_dynamics.ndim == 4
            else task_token_dynamics
        )
        if detach_shared:
            calibrated = calibrated.detach()
            scale = scale.detach()
        return self.severity_head(task_base, dynamics_base, calibrated, scale)

    def model_spec(self) -> dict:
        encoder_spec = self.encoder.model_spec() if hasattr(self.encoder, "model_spec") else {}
        return {
            "name": "femba_kan_mtl_v27",
            "encoder": encoder_spec,
            "encoder_trainable": False,
            "pooling": "token_mean",
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "kan_degree": self.kan_degree,
            "projection_variant": self.projection_variant,
            "projection_parameters": sum(
                parameter.numel() for parameter in self.shared_kan.parameters()
            ),
            "adaptive_basis_lr": getattr(self, "adaptive_basis_lr", None),
            "component_initialization": "separate_seed_streams_v1",
            "head_dropout": self.head_dropout_rate,
            "anchor_count": 4,
            "state_features": "absolute_calibrated_embedding + anchor_scale_normalized_rms_distance",
            "temporal_context": "mean of [t,t+1,t+2] embeddings within session",
            "severity_features": (
                "source-normalized frozen-FEMBA pooled dispersion, token velocity, and "
                "long-range drift progression plus a direct calibrated KAN residual"
            ),
            "state_head": "Linear(161,1)",
            "severity_head": "token-dynamics Linear(3,1) + bounded zero-initialized Linear(160,1) KAN residual",
            "severity_residual_bound": self.severity_head.residual_bound,
            "severity_base_normalization": "source-only frozen-FEMBA token-dynamics z-score",
        }
