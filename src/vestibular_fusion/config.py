from __future__ import annotations

import json
import os
from pathlib import Path


REQUIRED_PATH_KEYS = (
    "monifeixing_data_root",
    "vrq_data_root",
    "city_data_root",
    "pretrain_checkpoint",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_ROOT = PROJECT_ROOT / "reproducibility" / "protocols"


def native_path(value: str | Path) -> Path:
    text = os.fspath(value)
    if len(text) >= 3 and text[1:3] == ":\\":
        rest = text[3:].replace("\\", "/")
        return Path(f"/mnt/{text[0].lower()}/{rest}")
    if len(text) >= 3 and text[1:3] == ":/":
        return Path(f"/mnt/{text[0].lower()}/{text[3:]}")
    return Path(text).expanduser()


def load_config(path: str | Path, *, require_pretrain: bool = True) -> dict:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("paths"), dict):
        raise ValueError("Config must contain a top-level 'paths' object.")
    required = REQUIRED_PATH_KEYS if require_pretrain else REQUIRED_PATH_KEYS[:-1]
    missing = [key for key in required if key not in config["paths"]]
    if missing:
        raise ValueError(f"Config is missing required paths: {', '.join(missing)}")
    config["_config_path"] = str(config_path)
    config["paths"] = {key: native_path(value) for key, value in config["paths"].items()}
    config["protocol_root"] = (
        native_path(config["protocol_root"])
        if "protocol_root" in config
        else DEFAULT_PROTOCOL_ROOT
    )
    config["output_root"] = native_path(
        config.get("output_root", "outputs/v27_seed2001")
    )
    return config
