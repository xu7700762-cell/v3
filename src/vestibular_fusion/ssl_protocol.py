"""Independent SSL ablation contract; the published v27 protocol is unchanged."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import DEFAULT_PROTOCOL_ROOT
from .evaluation.io import read_json, sha256_file

SCHEMA = "femba_ssl_linear_probe_v1"
VARIANTS = {
    "A1": {"name": "Random-Frozen", "pretrained": False, "encoder_trainable": False},
    "A2": {"name": "Random-Scratch", "pretrained": False, "encoder_trainable": True},
    "A3": {"name": "Pretrained-Frozen", "pretrained": True, "encoder_trainable": False},
    "A4": {"name": "Pretrained-Finetune", "pretrained": True, "encoder_trainable": True},
}
DATASETS = ("monifeixing", "vrq", "city")
TASKS = ("state", "severity")
FOLDS = tuple(f"fold_{i}" for i in range(1, 6))
PROTOCOL_PATH = DEFAULT_PROTOCOL_ROOT / "femba_ssl_ablation.json"


def digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def experiment_protocol(seed: int) -> dict:
    if int(seed) <= 0:
        raise ValueError("Training seed must be positive")
    protocol = read_json(PROTOCOL_PATH)
    if protocol["schema"] != SCHEMA:
        raise ValueError("Unsupported SSL ablation protocol")
    return {**protocol, "training_seed": int(seed)}


def protocol_binding(protocol: dict, assets: dict, protocol_root: Path) -> dict:
    # Exclude checkpoint presence from the shared data identity: random-only runs
    # must remain comparable to pretrained-only runs without reading any weights.
    files = [f for f in assets["files"] if f["label"] != "pretrained FEMBA checkpoint"]
    return {
        "protocol_sha256": digest_json(protocol),
        "protocol_file_sha256": sha256_file(PROTOCOL_PATH),
        "bundle_sha256": sha256_file(Path(protocol_root) / "manifest.json"),
        "data_sha256": digest_json(sorted(
            [(f["label"], f["size"], f["sha256"]) for f in files]
        )),
    }


def split_identity(fold) -> dict:
    return {key: list(getattr(fold, key)) for key in (
        "source_subjects", "source_train_subjects", "source_val_subjects", "test_subjects"
    )}


def assert_empty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {path}")


def validate_metadata(payload: dict, expected: dict) -> None:
    mismatches = {k: {"expected": v, "actual": payload.get(k)}
                  for k, v in expected.items() if payload.get(k) != v}
    if mismatches:
        raise RuntimeError(f"SSL checkpoint/report metadata mismatch: {mismatches}")
