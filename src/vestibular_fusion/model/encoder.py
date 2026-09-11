from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except Exception as exc:  # pragma: no cover - exercised by preflight in WSL
    Mamba = None
    MAMBA_IMPORT_ERROR = exc
else:
    MAMBA_IMPORT_ERROR = None


PRETRAIN_MODEL_CONFIG = {
    "seq_length": 1280,
    "num_channels": 30,
    "exp": 4,
    "num_blocks": 4,
    "embed_dim": 35,
    "patch_size": (2, 16),
    "stride": (2, 16),
}


class MambaWrapper(nn.Module):
    def __init__(
        self,
        d_model: int,
        bidirectional: bool = True,
        bidirectional_strategy: Optional[str] = "add",
        **mamba_kwargs,
    ) -> None:
        super().__init__()
        if Mamba is None:
            raise ImportError("TemporalEncoder requires mamba-ssm.") from MAMBA_IMPORT_ERROR
        if bidirectional and bidirectional_strategy not in {"add", "ew_multiply"}:
            raise ValueError(f"Unsupported bidirectional strategy: {bidirectional_strategy}")
        self.bidirectional = bool(bidirectional)
        self.bidirectional_strategy = bidirectional_strategy
        self.mamba_fwd = Mamba(d_model=d_model, **mamba_kwargs)
        self.mamba_rev = Mamba(d_model=d_model, **mamba_kwargs) if bidirectional else None

    def forward(self, hidden_states, inference_params=None):
        output = self.mamba_fwd(hidden_states, inference_params=inference_params)
        if self.bidirectional:
            reverse = self.mamba_rev(
                hidden_states.flip(dims=(1,)), inference_params=inference_params
            ).flip(dims=(1,))
            output = output + reverse if self.bidirectional_strategy == "add" else output * reverse
        return output


class PatchEmbed(nn.Module):
    def __init__(self, inp_size, patch_size, stride, in_chans, embed_dim) -> None:
        super().__init__()
        self.grid_size = (
            (inp_size[0] - patch_size[0]) // stride[0] + 1,
            (inp_size[1] - patch_size[1]) // stride[1] + 1,
        )
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.proj(value.unsqueeze(1))
        value = value.reshape(value.shape[0], value.shape[1] * value.shape[2], value.shape[3])
        return value.permute(0, 2, 1)


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        seq_length: int = 1280,
        num_channels: int = 30,
        exp: int = 4,
        patch_size: Tuple[int, int] = (2, 16),
        stride: Tuple[int, int] = (2, 16),
        embed_dim: int = 35,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.seq_length = int(seq_length)
        self.num_channels = int(num_channels)
        self.exp = int(exp)
        self.patch_size = tuple(patch_size)
        self.stride = tuple(stride)
        self.embed_dim = int(embed_dim)
        self.num_blocks = int(num_blocks)
        self.patch_embed = PatchEmbed(
            (self.num_channels, self.seq_length), self.patch_size, self.stride, 1, self.embed_dim
        )
        grid_size = self.patch_embed.grid_size
        self.embedding_dim = grid_size[0] * self.embed_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size[1], self.embedding_dim))
        self.mamba_blocks = nn.ModuleList(
            [MambaWrapper(d_model=self.embedding_dim, expand=self.exp) for _ in range(self.num_blocks)]
        )
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(self.embedding_dim) for _ in range(self.num_blocks)]
        )

    def position_embedding_for(self, token_count: int) -> torch.Tensor:
        if token_count == self.pos_embed.shape[1]:
            return self.pos_embed
        if token_count < self.pos_embed.shape[1]:
            return self.pos_embed[:, :token_count]
        return F.interpolate(
            self.pos_embed.transpose(1, 2), size=token_count, mode="linear", align_corners=False
        ).transpose(1, 2)

    def forward_tokens(self, windows: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(windows)
        tokens = tokens + self.position_embedding_for(tokens.shape[1])
        for block, norm in zip(self.mamba_blocks, self.norm_layers):
            tokens = norm(tokens + block(tokens))
        return tokens

    def encode_windows(self, windows: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(windows).mean(dim=1)

    def model_spec(self) -> dict:
        return {
            "name": "four_block_bidirectional_temporal_encoder",
            "seq_length": self.seq_length,
            "num_channels": self.num_channels,
            "num_blocks": self.num_blocks,
            "embed_dim": self.embed_dim,
            "embedding_dim": self.embedding_dim,
            "patch_size": list(self.patch_size),
            "stride": list(self.stride),
            "token_shape": [80, 525],
            "bidirectional_strategy": "add",
        }


def build_encoder(
    device: torch.device, backend: str = "native", num_blocks: int | None = None
) -> TemporalEncoder:
    if backend != "native":
        raise ValueError("v1 supports only the native mamba-ssm backend")
    config = dict(PRETRAIN_MODEL_CONFIG)
    if num_blocks is not None:
        config["num_blocks"] = int(num_blocks)
    return TemporalEncoder(**config).to(device)


def load_pretrained_checkpoint(encoder: TemporalEncoder, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = encoder.state_dict()
    encoder_state = {}
    skipped = []
    remapped = 0
    for key, value in state_dict.items():
        clean = key[6:] if key.startswith("model.") else key
        mapped = clean.replace(".mamba_fwd.core.", ".mamba_fwd.").replace(
            ".mamba_rev.core.", ".mamba_rev."
        )
        remapped += int(mapped != clean)
        if mapped.startswith(("patch_embed", "pos_embed", "mamba_blocks", "norm_layers")):
            if mapped in model_state:
                encoder_state[mapped] = value
            else:
                skipped.append(mapped)
    result = encoder.load_state_dict(encoder_state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Incomplete Temporal Encoder checkpoint: "
            f"missing={list(result.missing_keys)}, unexpected={list(result.unexpected_keys)}"
        )
    return {
        "checkpoint": str(checkpoint_path),
        "loaded_keys": len(encoder_state),
        "remapped_keys": remapped,
        "skipped_keys": skipped,
        "missing_keys": [],
        "unexpected_keys": [],
    }
