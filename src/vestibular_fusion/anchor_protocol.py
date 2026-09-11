"""Independent validation-only anchor pilot contract."""
from pathlib import Path

from .config import DEFAULT_PROTOCOL_ROOT
from .evaluation.io import read_json, sha256_file
from .ssl_protocol import digest_json

SCHEMA = "femba_ssl_anchor_pilot_v2"
CONDITIONS = ("C0", "C1")
PROTOCOL_PATH = DEFAULT_PROTOCOL_ROOT / "femba_ssl_anchor_pilot.json"


def experiment_protocol(seed=2001):
    if seed <= 0:
        raise ValueError("Seed must be positive")
    value = read_json(PROTOCOL_PATH)
    if value["schema"] != SCHEMA:
        raise ValueError("Wrong anchor protocol schema")
    return {**value, "training_seed": seed}


def protocol_binding(protocol, assets, protocol_root):
    files = [f for f in assets["files"] if f["label"] != "pretrained FEMBA checkpoint"]
    return {"protocol_sha256": digest_json(protocol),
            "protocol_file_sha256": sha256_file(PROTOCOL_PATH),
            "bundle_sha256": sha256_file(Path(protocol_root) / "manifest.json"),
            "data_sha256": digest_json(sorted((f["label"], f["size"], f["sha256"]) for f in files))}
