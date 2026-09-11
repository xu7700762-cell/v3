from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("monifeixing", "vrq", "city")


def command_plan(config: Path, output_root: Path, device: str, include_mlp: bool) -> list[list[str]]:
    variants = ["fractional_dog_polykan"] + (["mlp"] if include_mlp else [])
    commands = [
        [sys.executable, "-m", "vestibular_fusion", "preflight", "--config", str(config)]
    ]
    for variant in variants:
        for dataset in DATASETS:
            commands.append(
                [
                    sys.executable,
                    "-m",
                    "vestibular_fusion",
                    "train",
                    "--config",
                    str(config),
                    "--dataset",
                    dataset,
                    "--training-seed",
                    "2001",
                    "--projection-variant",
                    variant,
                    "--device",
                    device,
                    "--output-root",
                    str(output_root / variant / "training" / dataset),
                ]
            )
            commands.append(
                [
                    sys.executable,
                    "-m",
                    "vestibular_fusion",
                    "evaluate",
                    "--config",
                    str(config),
                    "--dataset",
                    dataset,
                    "--checkpoint-root",
                    str(output_root / variant / "training" / dataset),
                    "--output-root",
                    str(output_root / variant / "evaluation" / dataset),
                    "--device",
                    device,
                ]
            )
        commands.append(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "verify_reproduction.py"),
                "--results-root",
                str(output_root / variant / "evaluation"),
                "--variant",
                variant,
            ]
        )
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce v27 with seed=2001")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reproduction_v27_seed2001",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-mlp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    plan = command_plan(config_path, output_root, args.device, args.include_mlp)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    os.environ["PYTHONHASHSEED"] = "2001"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from vestibular_fusion.config import load_config
    from vestibular_fusion.evaluate import run_evaluate
    from vestibular_fusion.preflight import run_preflight
    from vestibular_fusion.train import run_train

    config = load_config(config_path)
    run_preflight(config)
    variants = ["fractional_dog_polykan"] + (["mlp"] if args.include_mlp else [])
    for variant in variants:
        variant_config = {
            **config,
            "training_seed": 2001,
            "projection_variant": variant,
            "adaptive_basis_lr": 2e-4,
        }
        for dataset in DATASETS:
            training_root = output_root / variant / "training" / dataset
            evaluation_root = output_root / variant / "evaluation" / dataset
            run_train(variant_config, dataset, None, args.device, False, training_root)
            report = run_evaluate(
                variant_config, dataset, training_root, evaluation_root, args.device
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "verify_reproduction.py"),
                "--results-root",
                str(output_root / variant / "evaluation"),
                "--variant",
                variant,
            ],
            cwd=PROJECT_ROOT,
            check=True,
            env=os.environ.copy(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
