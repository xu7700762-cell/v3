from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .evaluate import run_evaluate
from .preflight import run_preflight
from .train import run_train


DATASETS = ("monifeixing", "vrq", "city")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vestibular_fusion")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "train", "evaluate"):
        sub.add_parser(name).add_argument("--config", required=True)
    train = sub.choices["train"]
    train.add_argument("--dataset", choices=DATASETS, required=True)
    train.add_argument("--fold", type=int)
    train.add_argument("--device", default="cuda")
    train.add_argument("--training-seed", type=int, default=2001)
    train.add_argument("--adaptive-basis-lr", type=float, default=2e-4)
    train.add_argument(
        "--projection-variant",
        choices=(
            "kan",
            "mlp",
            "fractional_dog_polykan",
        ),
        default="fractional_dog_polykan",
    )
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--output-root", type=Path)
    evaluate = sub.choices["evaluate"]
    evaluate.add_argument("--dataset", choices=DATASETS, required=True)
    evaluate.add_argument("--checkpoint-root", type=Path)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--output-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "train":
        if int(args.training_seed) <= 0:
            raise ValueError("--training-seed must be positive")
        config = {**config, "training_seed": int(args.training_seed)}
        config["projection_variant"] = str(args.projection_variant)
        if float(args.adaptive_basis_lr) <= 0.0:
            raise ValueError("--adaptive-basis-lr must be positive")
        config["adaptive_basis_lr"] = float(args.adaptive_basis_lr)
    if args.command == "preflight":
        run_preflight(config)
        return 0
    run_preflight(config)
    if args.command == "train":
        return int(
            run_train(
                config,
                args.dataset,
                args.fold,
                args.device,
                args.smoke,
                args.output_root.resolve() if args.output_root is not None else None,
            )
            or 0
        )
    if args.command == "evaluate":
        checkpoint_root = args.checkpoint_root or (
            config["output_root"] / "training" / args.dataset
        )
        output_root = args.output_root or (
            config["output_root"] / "evaluation" / args.dataset
        )
        report = run_evaluate(
            config, args.dataset, checkpoint_root, output_root, args.device
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(args.command)
