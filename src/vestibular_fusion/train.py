from __future__ import annotations

from pathlib import Path

from .training.runner import run_training


def run_train(
    config: dict,
    dataset: str,
    fold: int | None,
    device: str,
    smoke: bool,
    output_root: Path | None = None,
) -> int:
    if output_root is not None:
        root = Path(output_root)
    elif smoke:
        root = config["output_root"] / "smoke" / dataset / f"fold_{fold or 1}"
    elif fold is None:
        root = config["output_root"] / "training" / dataset
    else:
        root = config["output_root"] / "training" / dataset / f"fold_{fold}"
    result = run_training(config, dataset, fold, device, smoke, root)
    print(result)
    return 0
