from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from ..data.types import FeatureBank
from ..evaluation.io import read_json, sha256_file, write_json
from ..model.main import FEMBAKANMultiTaskModel


CACHE_SCHEMA = "femba_frozen_features_v2"


@dataclass(frozen=True)
class PooledFeatureCache:
    features: dict[str, np.ndarray]
    token_dynamics: dict[str, np.ndarray]
    metadata: dict
    cache_hit: bool
    load_or_build_seconds: float
    verification: dict

    def audit(self) -> dict:
        return {
            **self.metadata,
            "cache_hit": self.cache_hit,
            "load_or_build_seconds": self.load_or_build_seconds,
            "verification": self.verification,
        }


def fingerprint_feature_bank(bank: FeatureBank) -> str:
    digest = hashlib.sha256()
    for subject in sorted(bank.records):
        record = bank.records[subject]
        digest.update(subject.encode("utf-8"))
        for array in (record.windows, record.labels):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.shape).encode("ascii"))
            digest.update(contiguous.dtype.str.encode("ascii"))
            digest.update(memoryview(contiguous).cast("B"))
        digest.update(
            json.dumps(record.sessions, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _identity(
    config: dict, dataset: str, bank: FeatureBank, device: torch.device
) -> tuple[Path, dict]:
    pretrain_path = Path(config["paths"]["pretrain_checkpoint"])
    pretrain_sha = sha256_file(pretrain_path)
    data_fingerprint = fingerprint_feature_bank(bank)
    extraction_dtype = "float32"
    root = Path(
        config.get(
            "pooled_feature_cache_root",
            Path(config["output_root"]).parent / "femba_pooled_cache",
        )
    )
    stem = f"{pretrain_sha[:16]}-{data_fingerprint[:16]}-{extraction_dtype}"
    path = root / dataset / f"{stem}.npz"
    subject_keys = {
        subject: f"subject_{index:04d}"
        for index, subject in enumerate(sorted(bank.records))
    }
    metadata = {
        "cache_schema": CACHE_SCHEMA,
        "dataset": dataset,
        "cache_path": str(path),
        "pretrain_checkpoint_sha256": pretrain_sha,
        "data_fingerprint": data_fingerprint,
        "encoder_mode": "eval",
        "pooling": "token_mean",
        "token_shape": [80, 525],
        "feature_dim": 525,
        "token_dynamics_dim": 2,
        "token_dynamics": "adjacent-token RMS velocity + early/late token drift",
        "extraction_dtype": extraction_dtype,
        "subject_keys": subject_keys,
        "subject_lengths": {
            subject: int(len(bank.records[subject].windows))
            for subject in sorted(bank.records)
        },
    }
    return path, metadata


def _load(
    path: Path, metadata: dict, bank: FeatureBank
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        return None
    saved = read_json(metadata_path)
    if saved != metadata:
        return None
    with np.load(path, allow_pickle=False) as archive:
        features = {
            subject: np.asarray(archive[key], dtype=np.float32)
            for subject, key in metadata["subject_keys"].items()
        }
        token_dynamics = {
            subject: np.asarray(
                archive[f"{key}_token_dynamics"], dtype=np.float32
            )
            for subject, key in metadata["subject_keys"].items()
        }
    for subject, values in features.items():
        expected = (len(bank.records[subject].windows), 525)
        if values.shape != expected or not np.isfinite(values).all():
            raise RuntimeError(
                f"Invalid pooled FEMBA cache for {subject}: {values.shape}, expected {expected}"
            )
        dynamics = token_dynamics[subject]
        dynamics_expected = (len(bank.records[subject].windows), 2)
        if dynamics.shape != dynamics_expected or not np.isfinite(dynamics).all():
            raise RuntimeError(
                f"Invalid FEMBA token dynamics for {subject}: "
                f"{dynamics.shape}, expected {dynamics_expected}"
            )
    return features, token_dynamics


@torch.no_grad()
def _build(
    path: Path,
    metadata: dict,
    bank: FeatureBank,
    model: FEMBAKANMultiTaskModel,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    model.eval()
    features = {}
    token_dynamics = {}
    for subject in sorted(bank.records):
        windows = bank.records[subject].windows
        batches = []
        dynamics_batches = []
        for start in range(0, len(windows), int(batch_size)):
            tensor = torch.as_tensor(
                windows[start : start + int(batch_size)].astype(np.float32), device=device
            )
            pooled, dynamics = model.extract_frozen_features(tensor)
            batches.append(pooled.float().cpu().numpy())
            dynamics_batches.append(dynamics.float().cpu().numpy())
        values = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
        expected = (len(windows), 525)
        if values.shape != expected or not np.isfinite(values).all():
            raise RuntimeError(
                f"FEMBA cache extraction failed for {subject}: {values.shape}, expected {expected}"
            )
        features[subject] = values
        dynamics_values = np.concatenate(dynamics_batches, axis=0).astype(
            np.float32, copy=False
        )
        dynamics_expected = (len(windows), 2)
        if (
            dynamics_values.shape != dynamics_expected
            or not np.isfinite(dynamics_values).all()
        ):
            raise RuntimeError(
                f"FEMBA token dynamics extraction failed for {subject}: "
                f"{dynamics_values.shape}, expected {dynamics_expected}"
            )
        token_dynamics[subject] = dynamics_values
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b", dir=path.parent, suffix=".npz", delete=False
    ) as handle:
        np.savez(
            handle,
            **{
                metadata["subject_keys"][subject]: values
                for subject, values in features.items()
            },
            **{
                f"{metadata['subject_keys'][subject]}_token_dynamics": values
                for subject, values in token_dynamics.items()
            },
        )
        temporary = Path(handle.name)
    os.replace(temporary, path)
    write_json(path.with_suffix(".json"), metadata)
    return features, token_dynamics


def load_or_build_pooled_cache(
    config: dict,
    dataset: str,
    bank: FeatureBank,
    device: torch.device,
    model: FEMBAKANMultiTaskModel | None = None,
) -> PooledFeatureCache:
    started = time.perf_counter()
    path, metadata = _identity(config, dataset, bank, device)
    loaded = _load(path, metadata, bank)
    cache_hit = loaded is not None
    if loaded is None:
        if model is None:
            raise FileNotFoundError(
                f"Pooled FEMBA cache is missing and no extraction model was provided: {path}"
            )
        features, token_dynamics = _build(path, metadata, bank, model, device)
    else:
        features, token_dynamics = loaded
    if model is None:
        verification = {"status": "not_run", "reason": "no_extraction_model"}
    else:
        subjects = sorted(bank.records)
        raw = np.stack([bank.records[subject].windows[0] for subject in subjects]).astype(
            np.float32
        )
        expected_pooled = np.stack([features[subject][0] for subject in subjects]).astype(
            np.float32
        )
        expected_dynamics = np.stack(
            [token_dynamics[subject][0] for subject in subjects]
        ).astype(np.float32)
        with torch.no_grad():
            actual_pooled, actual_dynamics = model.extract_frozen_features(
                torch.as_tensor(raw, device=device)
            )
            actual_pooled = actual_pooled.float().cpu().numpy()
            actual_dynamics = actual_dynamics.float().cpu().numpy()
        pooled_max_abs_error = float(np.max(np.abs(actual_pooled - expected_pooled)))
        dynamics_max_abs_error = float(
            np.max(np.abs(actual_dynamics - expected_dynamics))
        )
        if max(pooled_max_abs_error, dynamics_max_abs_error) > 1e-5:
            raise RuntimeError(
                "Cached FEMBA features changed the frozen encoder output: "
                f"pooled={pooled_max_abs_error}, dynamics={dynamics_max_abs_error}"
            )
        verification = {
            "status": "passed",
            "subjects_checked": len(subjects),
            "windows_per_subject": 1,
            "pooled_max_abs_error": pooled_max_abs_error,
            "token_dynamics_max_abs_error": dynamics_max_abs_error,
            "absolute_tolerance": 1e-5,
        }
    return PooledFeatureCache(
        features=features,
        token_dynamics=token_dynamics,
        metadata=metadata,
        cache_hit=cache_hit,
        load_or_build_seconds=float(time.perf_counter() - started),
        verification=verification,
    )
