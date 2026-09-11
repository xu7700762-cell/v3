from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


DATASETS = ("monifeixing", "vrq", "city")
VARIANTS = {
    "fractional_dog_polykan": "fractional_dog_polykan",
    "mlp": "mlp",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def copy_csv(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8-sig", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {source}")
        rows = list(reader)
    with target.open("w", encoding="utf-8", newline="") as target_handle:
        writer = csv.DictWriter(
            target_handle, fieldnames=reader.fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def sanitized_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    return name if "." in name else "<local-path-redacted>"


def sanitize(value):
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    if isinstance(value, str) and (
        value.startswith(("/mnt/", "/home/"))
        or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        return sanitized_path(value)
    return value


def export_protocols(args: argparse.Namespace, output_root: Path) -> dict:
    protocol_root = output_root / "protocols"
    mono_report_source = Path(args.monifeixing_report)
    mono_predictions_source = Path(args.monifeixing_severity_predictions)
    mono_data_source = Path(args.monifeixing_data_manifest)
    vrq_source = Path(args.vrq_manifest)
    city_source = Path(args.city_manifest)

    mono_report = read_json(mono_report_source)
    write_json(
        protocol_root / "monifeixing" / "report.json",
        {
            "schema": "vestibular_fusion_monifeixing_protocol_v1",
            "identity_audit": sanitize(mono_report["identity_audit"]),
        },
    )

    labels = {}
    with mono_predictions_source.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            subject = str(row["subject_id"])
            label = int(row["y_true"])
            previous = labels.setdefault(subject, label)
            if previous != label:
                raise ValueError(f"Inconsistent monifeixing label for {subject}")
    label_path = protocol_root / "monifeixing" / "severity_labels.csv"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["subject_id", "y_true"], lineterminator="\n"
        )
        writer.writeheader()
        for subject in sorted(labels, key=lambda value: int(value.removeprefix("sub"))):
            writer.writerow({"subject_id": subject, "y_true": labels[subject]})

    mono_data = read_json(mono_data_source)
    write_json(
        protocol_root / "monifeixing" / "data_manifest.json",
        {
            "schema": "vestibular_fusion_monifeixing_data_v1",
            "inputs": {
                "source_mat_files": sanitize(
                    mono_data["inputs"]["source_mat_files"]
                )
            },
        },
    )

    vrq = read_json(vrq_source)
    write_json(
        protocol_root / "vrq" / "audit_manifest.json",
        {
            "schema": "vestibular_fusion_vrq_protocol_v1",
            "subject_protocols": sanitize(vrq["subject_protocols"]),
            "audit": {
                "read_only_qc": bool(vrq["audit"].get("read_only_qc", True)),
                "subjects": {
                    str(subject): {"ssq_label": int(metadata["ssq_label"])}
                    for subject, metadata in vrq["audit"]["subjects"].items()
                },
            },
            "folds": sanitize(vrq["folds"]),
            "run_fingerprint_payload": {
                "mat_key": str(vrq["run_fingerprint_payload"]["mat_key"]),
                "inputs": sanitize(vrq["run_fingerprint_payload"]["inputs"]),
            },
        },
    )

    city = read_json(city_source)
    city_audit = city["audit"]
    included_subjects = set(city_audit["included_subjects"])
    write_json(
        protocol_root / "city" / "audit_manifest.json",
        {
            "schema": "vestibular_fusion_city_protocol_v1",
            "audit": {
                "subjects": sanitize(
                    {
                        subject: metadata
                        for subject, metadata in city_audit["subjects"].items()
                        if subject in included_subjects
                    }
                ),
                "path_labels": sanitize(city_audit["path_labels"]),
                "inputs": sanitize(city_audit["inputs"]),
            },
            "fold_manifest": sanitize(city["fold_manifest"]),
        },
    )

    checkpoint = Path(args.pretrained_checkpoint)
    protocol_files = sorted(
        path
        for path in protocol_root.rglob("*")
        if path.is_file() and path != protocol_root / "manifest.json"
    )
    bundle = {
        "schema": "vestibular_fusion_protocol_bundle_v1",
        "checkpoint_schema": "femba_kan_mtl_v27",
        "training_seed": 2001,
        "split_seed": 42,
        "pretrained_femba": {
            "filename": str(args.release_asset_name),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
            "download_url": str(args.release_url),
            "tensor_count": 83,
        },
        "questionnaires": {
            "monifeixing_workbook_sha256": sha256_file(
                Path(args.monifeixing_workbook)
            ),
            "city_source_vrsq_workbook_sha256": sha256_file(
                Path(args.city_source_vrsq_workbook)
            ),
        },
        "source_artifacts": {
            "monifeixing_report_sha256": sha256_file(mono_report_source),
            "monifeixing_severity_predictions_sha256": sha256_file(
                mono_predictions_source
            ),
            "monifeixing_data_manifest_sha256": sha256_file(mono_data_source),
            "vrq_manifest_sha256": sha256_file(vrq_source),
            "city_manifest_sha256": sha256_file(city_source),
        },
        "protocol_files": {
            str(path.relative_to(output_root)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in protocol_files
        },
    }
    write_json(protocol_root / "manifest.json", bundle)
    return bundle


def metric_triplet(report: dict, task: str) -> dict:
    metrics = report[task]
    return {
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "auroc": float(metrics["AUROC"]),
    }


def export_reference_variant(
    source_root: Path, variant: str, reference_root: Path
) -> dict:
    datasets = {}
    for dataset in DATASETS:
        source = source_root / dataset
        report_path = source / "aggregate_report.json"
        state_path = source / "state_predictions.csv"
        severity_path = source / "severity_predictions.csv"
        for path in (report_path, state_path, severity_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing reference artifact: {path}")
        report = read_json(report_path)
        if report.get("status") != "complete" or len(report.get("folds", [])) != 5:
            raise ValueError(f"Incomplete five-fold reference report: {report_path}")
        variants = set()
        seeds = set()
        schemas = set()
        training_root = source_root.parent / "training" / dataset
        for fold in report["folds"]:
            fold_id = str(fold["fold_id"])
            training_report_path = training_root / fold_id / "report.json"
            checkpoint_path = training_root / fold_id / "checkpoint.pt"
            if not training_report_path.is_file() or not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Missing training provenance for {dataset}/{fold_id}"
                )
            training_report = read_json(training_report_path)
            variants.add(str(training_report.get("projection_variant")))
            seeds.add(int(training_report.get("training_seed")))
            schemas.add(str(training_report.get("checkpoint_schema")))
            if sha256_file(checkpoint_path) != str(fold["checkpoint_sha256"]):
                raise ValueError(
                    f"Checkpoint hash mismatch for {dataset}/{fold_id}"
                )
        if variants != {variant} or seeds != {2001} or schemas != {"femba_kan_mtl_v27"}:
            raise ValueError(
                f"Reference metadata mismatch for {dataset}: "
                f"variants={variants}, seeds={seeds}, schemas={schemas}"
            )
        target = reference_root / variant / dataset
        write_json(target / "aggregate_report.json", sanitize(report))
        target.mkdir(parents=True, exist_ok=True)
        copy_csv(state_path, target / "state_predictions.csv")
        copy_csv(severity_path, target / "severity_predictions.csv")
        datasets[dataset] = {
            "state": metric_triplet(report, "state_metrics"),
            "severity": metric_triplet(report, "severity_metrics"),
            "fold_checkpoint_sha256": [
                str(fold["checkpoint_sha256"]) for fold in report["folds"]
            ],
        }
    return datasets


def export_reference(args: argparse.Namespace, output_root: Path) -> dict:
    reference_root = output_root / "reference"
    variants = {
        "fractional_dog_polykan": export_reference_variant(
            Path(args.kan_evaluation_root),
            "fractional_dog_polykan",
            reference_root,
        ),
        "mlp": export_reference_variant(
            Path(args.mlp_evaluation_root), "mlp", reference_root
        ),
    }
    expected = {
        "schema": "vestibular_fusion_reference_metrics_v1",
        "checkpoint_schema": "femba_kan_mtl_v27",
        "training_seed": 2001,
        "split_seed": 42,
        "metric_tolerance_percentage_points": 1.0,
        "variants": variants,
    }
    write_json(reference_root / "expected_metrics.json", expected)
    files = sorted(
        path
        for path in reference_root.rglob("*")
        if path.is_file() and path != reference_root / "manifest.json"
    )
    manifest = {
        "schema": "vestibular_fusion_reference_bundle_v1",
        "checkpoint_schema": "femba_kan_mtl_v27",
        "training_seed": 2001,
        "files": {
            str(path.relative_to(output_root)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
    }
    write_json(reference_root / "manifest.json", manifest)
    return expected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--monifeixing-report", required=True)
    parser.add_argument("--monifeixing-severity-predictions", required=True)
    parser.add_argument("--monifeixing-data-manifest", required=True)
    parser.add_argument("--vrq-manifest", required=True)
    parser.add_argument("--city-manifest", required=True)
    parser.add_argument("--monifeixing-workbook", required=True)
    parser.add_argument("--city-source-vrsq-workbook", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--kan-evaluation-root", required=True)
    parser.add_argument("--mlp-evaluation-root", required=True)
    parser.add_argument(
        "--release-asset-name", default="pretrained_femba_v27.ckpt"
    )
    parser.add_argument(
        "--release-url",
        default=(
            "https://github.com/xu7700762-cell/v2/releases/download/"
            "v27-repro-assets/pretrained_femba_v27.ckpt"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    export_protocols(args, output_root)
    export_reference(args, output_root)
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
