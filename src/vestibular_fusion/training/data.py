from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

from ..data import city, monifeixing, vrq
from ..data.types import FeatureBank
from ..evaluation.io import read_csv, read_json
from ..evaluation.metrics import subject_sort_key


@dataclass(frozen=True)
class SeverityExample:
    subject_id: str
    reference_session: str
    task_session: str
    label: int


@dataclass(frozen=True)
class FoldProtocol:
    source_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    source_train_subjects: tuple[str, ...]
    source_val_subjects: tuple[str, ...]
    source_examples: tuple[SeverityExample, ...]
    test_examples: tuple[SeverityExample, ...]

    def __post_init__(self) -> None:
        source = set(self.source_subjects)
        test = set(self.test_subjects)
        train = set(self.source_train_subjects)
        val = set(self.source_val_subjects)
        if source & test:
            raise ValueError("Source and outer-test subjects must be identity-disjoint")
        if train & val:
            raise ValueError("Source-train and source-val subjects must be disjoint")
        if train | val != source:
            raise ValueError("Source-train/source-val must partition outer source")


@dataclass
class TrainingDataset:
    bank: FeatureBank
    folds: dict[str, FoldProtocol]


def _select_examples(
    examples: list[SeverityExample], subjects: list[str] | tuple[str, ...]
) -> tuple[SeverityExample, ...]:
    selected = set(subjects)
    return tuple(example for example in examples if example.subject_id in selected)


def _load_monifeixing(config: dict) -> TrainingDataset:
    root = Path(config["protocol_root"]) / "monifeixing"
    report = read_json(root / "report.json")
    labels = {
        str(row["subject_id"]): int(row["y_true"])
        for row in read_csv(root / "severity_labels.csv")
    }
    examples = [
        SeverityExample(subject, "rest1", "rest2", labels[subject])
        for subject in sorted(labels, key=subject_sort_key)
    ]
    folds = {}
    for fold_id, split in report["identity_audit"]["folds"].items():
        source = tuple(str(subject) for subject in split["source_outer_train_subjects"])
        test = tuple(str(subject) for subject in split["test_subjects"])
        folds[fold_id] = FoldProtocol(
            source_subjects=source,
            test_subjects=test,
            source_train_subjects=tuple(str(subject) for subject in split["source_train_subjects"]),
            source_val_subjects=tuple(str(subject) for subject in split["source_val_subjects"]),
            source_examples=_select_examples(examples, source),
            test_examples=_select_examples(examples, test),
        )
    return TrainingDataset(
        bank=monifeixing.build_raw_bank(Path(config["paths"]["monifeixing_data_root"])),
        folds=folds,
    )


def _load_vrq(config: dict) -> TrainingDataset:
    root = Path(config["protocol_root"]) / "vrq"
    manifest = read_json(root / "audit_manifest.json")
    protocols = [vrq.SubjectProtocol(**row) for row in manifest["subject_protocols"]]
    task_sessions = {row.subject_id: row.final_task for row in protocols}
    examples = [
        SeverityExample(subject, "rest01", task_sessions[subject], int(metadata["ssq_label"]))
        for subject, metadata in sorted(
            manifest["audit"]["subjects"].items(), key=lambda item: subject_sort_key(item[0])
        )
        if subject in task_sessions
    ]
    folds = {}
    for fold_id, split in manifest["folds"].items():
        source = tuple(
            sorted(split["train_subjects"] + split["val_subjects"], key=subject_sort_key)
        )
        test = tuple(str(subject) for subject in split["test_subjects"])
        folds[fold_id] = FoldProtocol(
            source_subjects=source,
            test_subjects=test,
            source_train_subjects=tuple(str(subject) for subject in split["train_subjects"]),
            source_val_subjects=tuple(str(subject) for subject in split["val_subjects"]),
            source_examples=_select_examples(examples, source),
            test_examples=_select_examples(examples, test),
        )
    payload = manifest["run_fingerprint_payload"]
    return TrainingDataset(
        bank=vrq.build_raw_bank(
            Path(config["paths"]["vrq_data_root"]), payload["mat_key"], manifest["audit"], protocols
        ),
        folds=folds,
    )


def _load_city(config: dict) -> TrainingDataset:
    root = Path(config["protocol_root"]) / "city" / "audit_manifest.json"
    manifest = read_json(root)
    audit = copy.deepcopy(manifest["audit"])
    aliases = {}
    for subject, metadata in audit["subjects"].items():
        for segment in metadata.get("segments", []):
            if segment.get("path_score") is not None:
                aliases[(subject, int(segment["route_order"]))] = city.session_alias(
                    segment, metadata["anchor_session"]
                )
    examples = [
        SeverityExample(
            str(row["subject_id"]),
            "rest01",
            aliases[(str(row["subject_id"]), int(row["route_order"]))],
            int(row["path_label"]),
        )
        for row in audit["path_labels"]
    ]
    folds = {}
    for fold_id, split in manifest["fold_manifest"]["folds"].items():
        source = tuple(
            sorted(split["train_subjects"] + split["val_subjects"], key=subject_sort_key)
        )
        test = tuple(str(subject) for subject in split["test_subjects"])
        folds[fold_id] = FoldProtocol(
            source_subjects=source,
            test_subjects=test,
            source_train_subjects=tuple(str(subject) for subject in split["train_subjects"]),
            source_val_subjects=tuple(str(subject) for subject in split["val_subjects"]),
            source_examples=_select_examples(examples, source),
            test_examples=_select_examples(examples, test),
        )
    return TrainingDataset(
        bank=city.build_raw_bank(Path(config["paths"]["city_data_root"]), audit),
        folds=folds,
    )


def load_training_dataset(config: dict, dataset: str) -> TrainingDataset:
    if dataset == "monifeixing":
        return _load_monifeixing(config)
    if dataset == "vrq":
        return _load_vrq(config)
    if dataset == "city":
        return _load_city(config)
    raise ValueError(f"Unsupported dataset: {dataset}")
