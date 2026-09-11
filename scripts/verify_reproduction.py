from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from vestibular_fusion.evaluation.metrics import binary_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = PROJECT_ROOT / "reproducibility"
DATASETS = ("monifeixing", "vrq", "city")
VARIANTS = ("fractional_dog_polykan", "mlp")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_reference_manifest() -> None:
    manifest = read_json(REPRO_ROOT / "reference" / "manifest.json")
    for relative, metadata in manifest["files"].items():
        path = REPRO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing bundled reference file: {path}")
        actual = sha256_file(path)
        if actual != metadata["sha256"]:
            raise RuntimeError(
                f"Bundled reference SHA-256 mismatch for {relative}: {actual}"
            )


def metric_triplet(report: dict, task: str) -> dict:
    values = report[task]
    return {
        "accuracy": float(values["accuracy"]),
        "balanced_accuracy": float(values["balanced_accuracy"]),
        "auroc": float(values["AUROC"]),
    }


def verify_predictions(root: Path, report: dict) -> None:
    for filename, task in (
        ("state_predictions.csv", "state_metrics"),
        ("severity_predictions.csv", "severity_metrics"),
    ):
        rows = read_csv(root / filename)
        if filename == "state_predictions.csv":
            rows = [row for row in rows if not int(row["calibration_anchor"])]
        recalculated = binary_metrics(rows)
        recorded = report[task]
        for key in ("accuracy", "balanced_accuracy", "AUROC"):
            if not math.isclose(
                float(recalculated[key]), float(recorded[key]), rel_tol=0.0, abs_tol=1e-12
            ):
                raise RuntimeError(
                    f"{root}/{filename} does not reproduce {task}.{key}"
                )


def verify_variant(results_root: Path, variant: str, expected: dict) -> dict:
    tolerance = float(expected["metric_tolerance_percentage_points"])
    summary = {}
    for dataset in DATASETS:
        root = results_root / dataset
        report = read_json(root / "aggregate_report.json")
        if report.get("status") != "complete" or len(report.get("folds", [])) != 5:
            raise RuntimeError(f"Incomplete five-fold report: {root}")
        seeds = {int(fold.get("training_seed", -1)) for fold in report["folds"]}
        variants = {str(fold.get("projection_variant")) for fold in report["folds"]}
        if seeds != {2001} or variants != {variant}:
            raise RuntimeError(
                f"Wrong v27 provenance for {dataset}: seeds={seeds}, variants={variants}"
            )
        verify_predictions(root, report)
        summary[dataset] = {}
        for task, report_key in (("state", "state_metrics"), ("severity", "severity_metrics")):
            actual = metric_triplet(report, report_key)
            target = expected["variants"][variant][dataset][task]
            differences = {
                key: abs(actual[key] - float(target[key])) * 100.0 for key in actual
            }
            if max(differences.values()) > tolerance + 1e-12:
                raise RuntimeError(
                    f"{variant}/{dataset}/{task} exceeds {tolerance:.2f} pp: {differences}"
                )
            summary[dataset][task] = {
                "metrics": actual,
                "max_difference_pp": max(differences.values()),
            }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify v27 seed=2001 results")
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    expected = read_json(REPRO_ROOT / "reference" / "expected_metrics.json")
    verify_reference_manifest()
    if args.results_root is not None:
        if args.variant is None:
            parser.error("--variant is required with --results-root")
        roots = {args.variant: args.results_root.expanduser().resolve()}
    else:
        roots = {
            variant: REPRO_ROOT / "reference" / variant for variant in VARIANTS
        }
    result = {
        "status": "passed",
        "checkpoint_schema": expected["checkpoint_schema"],
        "training_seed": expected["training_seed"],
        "variants": {
            variant: verify_variant(root, variant, expected)
            for variant, root in roots.items()
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
