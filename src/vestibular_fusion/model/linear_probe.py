from __future__ import annotations

from contextlib import nullcontext
import hashlib
from pathlib import Path

import torch
from torch import nn

from .encoder import TemporalEncoder, load_pretrained_checkpoint
from ..evaluation.io import sha256_file
from ..ssl_protocol import VARIANTS


def tensor_state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FEMBALinearProbe(nn.Module):
    """Identical encoder + one affine head in every cell of the 2x2 experiment."""

    def __init__(self, encoder: nn.Module, *, encoder_trainable: bool, seed: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_trainable = bool(encoder_trainable)
        self.encoder.requires_grad_(self.encoder_trainable)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 1)
            self.head = nn.Linear(525, 1, bias=True)
        self.train()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.train(bool(mode) and self.encoder_trainable)
        return self

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim not in (3, 4) or windows.shape[-2:] != (30, 1280):
            raise ValueError(f"Expected [B,30,1280] or [B,11,30,1280]: {windows.shape}")
        if windows.ndim == 4 and windows.shape[1] != 11:
            raise ValueError("Severity requires exactly 11 locked task windows")
        with nullcontext() if self.encoder_trainable else torch.no_grad():
            tokens = self.encoder.forward_tokens(windows.reshape(-1, 30, 1280).float())
            if tokens.shape[1:] != (80, 525):
                raise ValueError(f"Unexpected FEMBA tokens: {tokens.shape}")
            pooled = tokens.float().mean(dim=1)
            if windows.ndim == 4:
                pooled = pooled.reshape(windows.shape[0], 11, 525).mean(dim=1)
        return self.head(pooled).squeeze(-1)


def build_probe(
    variant: str, seed: int, device: torch.device, *,
    checkpoint_path: Path | None = None, expected_sha256: str | None = None,
    encoder_factory=TemporalEncoder,
) -> tuple[FEMBALinearProbe, dict]:
    spec = VARIANTS[variant]
    # Build on CPU so encoder/head initialization cannot consume the data RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        encoder = encoder_factory()
    load_info = None
    checkpoint_sha256 = None
    if spec["pretrained"]:
        if checkpoint_path is None or expected_sha256 is None:
            raise ValueError("Pretrained variants require the official checkpoint and SHA-256")
        checkpoint_sha256 = sha256_file(Path(checkpoint_path))
        if checkpoint_sha256 != expected_sha256:
            raise RuntimeError("Official pretrained checkpoint SHA-256 mismatch")
        load_info = load_pretrained_checkpoint(encoder, Path(checkpoint_path))
        if load_info["loaded_keys"] != 83 or any(load_info[key] for key in (
            "missing_keys", "unexpected_keys", "skipped_keys"
        )):
            raise RuntimeError(f"Incomplete pretrained encoder: {load_info}")
    model = FEMBALinearProbe(encoder, encoder_trainable=spec["encoder_trainable"], seed=seed)
    initial = {
        "encoder_sha256": tensor_state_hash(model.encoder.state_dict()),
        "head_sha256": tensor_state_hash(model.head.state_dict()),
        "pretrain_checkpoint_sha256": checkpoint_sha256,
        "pretrain_load_info": load_info,
        "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
        "head_parameters": sum(p.numel() for p in model.head.parameters()),
        "parameter_shapes": {n: list(p.shape) for n, p in model.named_parameters()},
    }
    return model.to(device), initial
