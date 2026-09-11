from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from vestibular_fusion.config import DEFAULT_PROTOCOL_ROOT, load_config
from vestibular_fusion.evaluation.io import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_minimal_config_uses_bundled_protocols(tmp_path: Path):
    path = tmp_path / "paths.json"
    path.write_text(
        json.dumps(
            {
                "paths": {
                    "monifeixing_data_root": "mono",
                    "vrq_data_root": "vrq",
                    "city_data_root": "city",
                    "pretrain_checkpoint": "pretrained.ckpt",
                }
            }
        ),
        encoding="utf-8",
    )
    config = load_config(path)
    assert config["protocol_root"] == DEFAULT_PROTOCOL_ROOT
    assert "asset_root" not in config["paths"]


def test_bundled_protocol_manifest_hashes_are_valid():
    manifest = json.loads(
        (ROOT / "reproducibility" / "protocols" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "protocols/manifest.json" not in manifest["protocol_files"]
    for relative, metadata in manifest["protocol_files"].items():
        assert sha256_file(ROOT / "reproducibility" / relative) == metadata["sha256"]


def test_reference_bundle_recalculates_and_verifies():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_reproduction.py")],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "passed"
    assert result["training_seed"] == 2001
    assert set(result["variants"]) == {"fractional_dog_polykan", "mlp"}


def test_reproduction_dry_run_covers_three_datasets(tmp_path: Path):
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "reproduce_v27_seed2001.py"),
            "--config",
            str(tmp_path / "paths.json"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)
    text = json.dumps(plan)
    assert all(dataset in text for dataset in ("monifeixing", "vrq", "city"))
    assert "fractional_dog_polykan" in text
    assert "mlp" not in text
