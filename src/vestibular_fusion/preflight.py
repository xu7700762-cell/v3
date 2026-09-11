from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from .evaluation.io import read_json, sha256_file
from .protocol import PROTOCOL


EXPECTED_PACKAGES = {
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "scikit-learn": "1.7.2",
    "openpyxl": "3.1.5",
    "joblib": "1.5.3",
    "torch": "2.11.0+cu128",
    "mamba-ssm": "2.3.1",
}


def _require(path: Path, label: str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return {
        "label": label,
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _require_hash(path: Path, expected: str, label: str) -> dict:
    result = _require(path, label)
    if result["sha256"] != str(expected):
        raise RuntimeError(
            f"SHA-256 mismatch for {label}: expected {expected}, found {result['sha256']}"
        )
    return result


def _check_environment() -> dict:
    if tuple(sys.version_info[:2]) != (3, 10):
        raise RuntimeError(f"Python 3.10 is required; found {platform.python_version()}")
    if platform.system() != "Linux":
        raise RuntimeError("Training must run under WSL2 Linux")
    versions = {}
    for distribution, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Missing required package {distribution}=={expected}") from exc
        if actual != expected:
            raise RuntimeError(f"{distribution}=={expected} is required; found {actual}")
        versions[distribution] = actual
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full FEMBA fine-tuning")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA bfloat16 AMP is required for stable FEMBA fine-tuning")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "amp_dtype": "bfloat16",
    }


def _test_subject_key(fold: dict) -> str:
    for key in ("test_subjects", "vrq_test_subjects"):
        if key in fold:
            return key
    raise RuntimeError("A fold has no test subject list")


def _check_fivefold_subject_split(
    manifest: dict, label: str, *, fold_key: str = "folds"
) -> None:
    folds = manifest.get(fold_key)
    if not isinstance(folds, dict) or set(folds) != {
        f"fold_{index}" for index in range(1, 6)
    }:
        raise RuntimeError(f"{label} must contain exactly fold_1..fold_5")
    test_sets = []
    for fold_id, fold in sorted(folds.items()):
        test_key = _test_subject_key(fold)
        test = set(fold[test_key])
        for train_key, val_key in (
            ("train_subjects", "val_subjects"),
            ("source_train_subjects", "source_val_subjects"),
        ):
            if train_key in fold and val_key in fold:
                overlap = set(fold[train_key]) & set(fold[val_key])
                if overlap:
                    raise RuntimeError(
                        f"{label}/{fold_id} source-train/source-val subjects overlap: "
                        f"{sorted(overlap)}"
                    )
        source_groups = []
        if "source_outer_train_subjects" in fold:
            source_groups.append(set(fold["source_outer_train_subjects"]))
        else:
            source_groups.extend(
                set(fold[key]) for key in ("train_subjects", "val_subjects") if key in fold
            )
        if any(test & source for source in source_groups):
            raise RuntimeError(f"{label}/{fold_id} has source/test subject overlap")
        test_sets.append((fold_id, test))
    for index, (left_id, left) in enumerate(test_sets):
        for right_id, right in test_sets[index + 1 :]:
            overlap = left & right
            if overlap:
                raise RuntimeError(
                    f"{label} test subjects overlap between {left_id} and {right_id}: {sorted(overlap)}"
                )


def _summary_digest(files: list[dict]) -> str:
    payload = "\n".join(
        f"{item['label']}\0{item['size']}\0{item['sha256']}"
        for item in sorted(files, key=lambda item: item["label"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_source_tree() -> dict:
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parents[1]
    removed = [
        package_root / "reproduce.py",
        package_root / "verify.py",
        package_root / "evaluation" / "fusion.py",
        package_root / "evaluation" / "geometry.py",
        package_root / "evaluation" / "context.py",
        package_root / "evaluation" / "runner.py",
        package_root / "evaluation" / "severity.py",
        package_root / "model" / "a1.py",
        package_root / "model" / "severity.py",
    ]
    existing = [str(path) for path in removed if path.exists()]
    if existing:
        raise RuntimeError(f"Removed downstream modules still exist: {existing}")
    return {
        "package_root": str(package_root),
        "python_files": len(list(package_root.rglob("*.py"))),
        "removed_downstream_modules": len(removed),
    }


def _check_protocol(config: dict, *, require_pretrain: bool = True) -> dict:
    paths = config["paths"]
    protocol_root = Path(config["protocol_root"])
    bundle_path = protocol_root / "manifest.json"
    bundle = read_json(bundle_path)
    if bundle.get("checkpoint_schema") != "femba_kan_mtl_v27":
        raise RuntimeError("Bundled protocol checkpoint schema is not v27")
    if int(bundle.get("training_seed", -1)) != 2001:
        raise RuntimeError("Bundled protocol training seed is not 2001")
    files = [_require(bundle_path, "bundled protocol manifest")]
    for relative, metadata in bundle["protocol_files"].items():
        path = protocol_root.parent / relative
        files.append(
            _require_hash(path, metadata["sha256"], f"bundled protocol {relative}")
        )
    mono_root = protocol_root / "monifeixing"
    mono_report = read_json(mono_root / "report.json")
    _check_fivefold_subject_split(mono_report["identity_audit"], "monifeixing report")
    mono_outer = read_json(mono_root / "data_manifest.json")
    vrq_manifest = read_json(protocol_root / "vrq" / "audit_manifest.json")
    city_manifest = read_json(protocol_root / "city" / "audit_manifest.json")
    _check_fivefold_subject_split(vrq_manifest, "VRQ audit manifest")
    _check_fivefold_subject_split(city_manifest["fold_manifest"], "city fold manifest")
    pretrain_expected = bundle["pretrained_femba"]["sha256"]
    if require_pretrain:
        files.append(
            _require_hash(
                Path(paths["pretrain_checkpoint"]), pretrain_expected, "pretrained FEMBA checkpoint"
            )
        )
    for name, entry in mono_outer["inputs"]["source_mat_files"].items():
        files.append(
            _require_hash(
                Path(paths["monifeixing_data_root"]) / name,
                entry["sha256"],
                f"monifeixing data {name}",
            )
        )
    vrq_inputs = vrq_manifest["run_fingerprint_payload"]["inputs"]
    for name, expected in vrq_inputs["mat_sha256"].items():
        files.append(
            _require_hash(Path(paths["vrq_data_root"]) / name, expected, f"VRQ data {name}")
        )
    for subject, metadata in city_manifest["audit"]["subjects"].items():
        name = Path(metadata["mat_path"]).name
        files.append(
            _require_hash(
                Path(paths["city_data_root"]) / name,
                metadata["mat_sha256"],
                f"city data {subject}",
            )
        )
    return {
        "protocol_root": str(protocol_root),
        "file_count": len(files),
        "summary_sha256": _summary_digest(files),
        "files": files,
    }


def run_preflight(config: dict) -> dict:
    projection_variant = str(
        config.get("projection_variant", "fractional_dog_polykan")
    )
    if projection_variant not in {
        "kan",
        "mlp",
        "fractional_dog_polykan",
    }:
        raise ValueError(
            "projection_variant must be 'kan', 'mlp', or "
            "'fractional_dog_polykan'"
        )
    shared_heads = {
        "kan": PROTOCOL["shared_head"],
        "mlp": "ParameterMatchedMLP(525,245,160) with anchor-relative state and token-dynamics PRISM severity",
        "fractional_dog_polykan": (
            "DualRateFractionalDoGPolynomialKAN(525,160,degree=2,featurewise_DoG) "
            "with anchor-relative state and token-dynamics PRISM severity"
        ),
    }
    protocol = {
        **PROTOCOL,
        "training_seed": int(config.get("training_seed", PROTOCOL["training_seed"])),
        "projection_variant": projection_variant,
        "adaptive_basis_lr": float(config.get("adaptive_basis_lr", 2e-4)),
        "shared_head": shared_heads[projection_variant],
    }
    result = {
        "status": "passed",
        "protocol": protocol,
        "environment": _check_environment(),
        "source": _check_source_tree(),
        "assets": _check_protocol(config),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
