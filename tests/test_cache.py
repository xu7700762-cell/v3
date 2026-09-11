from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from vestibular_fusion.data.types import AuditMetadata, FeatureBank, SubjectRecord
from vestibular_fusion.training.cache import (
    CACHE_SCHEMA,
    fingerprint_feature_bank,
    load_or_build_pooled_cache,
)


class FakePoolingModel(nn.Module):
    def extract_frozen_features(
        self, windows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = windows.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        pooled = values.expand(-1, 525)
        dynamics = torch.cat([values, values + 1.0], dim=-1)
        return pooled, dynamics


def _bank(value: float = 0.0) -> FeatureBank:
    windows = np.full((4, 30, 1280), value, dtype=np.float32)
    return FeatureBank(
        records={
            "s1": SubjectRecord(
                windows=windows,
                labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
                sessions=["rest", "rest", "task", "task"],
            )
        },
        samples=[],
        audit=AuditMetadata({}),
    )


def test_pooled_cache_is_reused_and_bound_to_raw_data(tmp_path: Path):
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"pretrained")
    config = {
        "output_root": str(tmp_path / "outputs" / "seed42"),
        "paths": {"pretrain_checkpoint": str(checkpoint)},
    }
    first = load_or_build_pooled_cache(
        config,
        "synthetic",
        _bank(),
        torch.device("cpu"),
        FakePoolingModel(),
    )
    assert not first.cache_hit
    assert first.metadata["cache_schema"] == CACHE_SCHEMA
    assert first.features["s1"].shape == (4, 525)
    assert first.token_dynamics["s1"].shape == (4, 2)

    second = load_or_build_pooled_cache(
        config, "synthetic", _bank(), torch.device("cpu")
    )
    assert second.cache_hit
    assert np.array_equal(first.features["s1"], second.features["s1"])
    assert np.array_equal(
        first.token_dynamics["s1"], second.token_dynamics["s1"]
    )

    assert fingerprint_feature_bank(_bank()) != fingerprint_feature_bank(_bank(1.0))
    with pytest.raises(FileNotFoundError, match="cache is missing"):
        load_or_build_pooled_cache(
            config, "synthetic", _bank(1.0), torch.device("cpu")
        )
