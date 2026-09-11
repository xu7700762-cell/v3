"""Frozen FEMBA readouts; reference metadata is deliberately absent from the API."""
import torch
from torch import nn

from .kan import FractionalDoGPolynomialKANLayer, PolynomialKANLayer
from .main import ParameterMatchedMLP
from .linear_probe import tensor_state_hash

CONFIGS = tuple(f"{r}_{h}" for r in ("R0", "R1", "R2")
                for h in ("dog", "mlp", "poly")) + (
    "linear", "R2_order1", "R2_no_dog", "R2_order1_no_dog")


def statistics(x, dim):
    mean = x.mean(dim)
    std = ((x - mean.unsqueeze(dim)).square().mean(dim) + 1e-6).sqrt()
    return mean, std


def window_summary(x):
    mean, std = statistics(x, -2)
    delta = (x[..., 1:, :] - x[..., :-1, :]).abs().mean(-2)
    return torch.cat((mean, std, delta), -1)


def task_summary(x):
    if x.shape[-2] != 11:
        raise ValueError("Severity requires exactly eleven ordered windows")
    mean, std = statistics(x, -2)
    return torch.cat((mean, std, x[..., -3:, :].mean(-2) - x[..., :3, :].mean(-2)), -1)


class TokenReadout(nn.Module):
    def __init__(self, config, task, seed=2001):
        super().__init__()
        if config not in CONFIGS or task not in ("state", "severity"):
            raise ValueError("Unknown locked token readout")
        self.config, self.task = config, task
        self.representation = config.split("_")[0]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 101)
            if config != "linear":
                kind = config.split("_", 1)[1]
                projection = (ParameterMatchedMLP(525, 160, 246) if kind == "mlp" else
                              PolynomialKANLayer(525, 160, 2) if kind == "poly" else
                              FractionalDoGPolynomialKANLayer(525, 160, 2))
                self.mapping = nn.Sequential(nn.LayerNorm(525), projection, nn.LayerNorm(160))
                if "order1" in config:
                    projection.fractional_order_logit.requires_grad_(False)
                if "no_dog" in config:
                    for name in ("dog_mix_logits", "dog_scale_logit", "dog_shift"):
                        getattr(projection, name).requires_grad_(False)
            width = 525 if config == "linear" else (480 if task == "state" else 1440) if self.representation == "R2" else 160
            torch.manual_seed(seed + 211)
            self.classifier = nn.Linear(width, 1)

    def features(self, tokens):
        expected_ndim = 3 if self.task == "state" else 4
        if tokens.ndim != expected_ndim or tokens.shape[-2:] != (80, 525):
            raise ValueError("Expected state [B,80,525] or severity [B,11,80,525] tokens")
        if self.task == "severity" and tokens.shape[1] != 11:
            raise ValueError("Severity requires eleven windows")
        tokens = tokens.float()
        if self.config == "linear" or self.representation == "R0":
            pooled = tokens.mean(-2)
            if self.task == "severity":
                pooled = pooled.mean(1)
            return pooled if self.config == "linear" else self.mapping(pooled)
        mapped = self.mapping(tokens)
        if self.representation == "R1":
            pooled = mapped.mean(-2)
            return pooled.mean(1) if self.task == "severity" else pooled
        pooled = window_summary(mapped)
        return task_summary(pooled) if self.task == "severity" else pooled

    def forward(self, tokens):
        # The full readout, including polynomial linear projections, is float32.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            return self.classifier(self.features(tokens)).squeeze(-1)

    def initialization(self):
        return {"head_sha256": tensor_state_hash(self.state_dict()),
                "mapping_sha256": tensor_state_hash(self.mapping.state_dict()) if hasattr(self, "mapping") else None,
                "output_sha256": tensor_state_hash(self.classifier.state_dict()),
                "parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "shapes": {n: list(p.shape) for n, p in self.named_parameters()}}


class FrozenTokenProbe(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder, self.head = encoder.requires_grad_(False).eval(), head

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, windows):
        if windows.shape[-2:] != (30, 1280):
            raise ValueError("Expected raw five-second EEG windows")
        with torch.no_grad(), torch.autocast(device_type=windows.device.type, dtype=torch.bfloat16,
                                            enabled=windows.device.type == "cuda"):
            tokens = self.encoder.forward_tokens(windows.reshape(-1, 30, 1280).float()).float()
        if windows.ndim == 4:
            tokens = tokens.reshape(len(windows), 11, 80, 525)
        return self.head(tokens)
