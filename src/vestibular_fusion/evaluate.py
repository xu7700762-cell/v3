from __future__ import annotations

from pathlib import Path

from .evaluation.trained import run_trained_evaluation


def run_evaluate(
    config: dict,
    dataset: str,
    checkpoint_root: Path,
    output_root: Path,
    device: str,
) -> dict:
    return run_trained_evaluation(
        config,
        dataset,
        Path(checkpoint_root),
        Path(output_root),
        device,
    )
