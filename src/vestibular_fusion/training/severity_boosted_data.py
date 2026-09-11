"""City source-only paths with continuous scores for nested severity development."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..evaluation.io import read_json
from ..ssl_protocol import digest_json
from .token_data import load_source_data, sample_hash


@dataclass(frozen=True)
class ContinuousSeverityExample:
    subject_id: str
    session: str
    indices: tuple
    label: int
    sample_id: str
    path_score: float

    def identity(self):
        return {"subject_id": self.subject_id, "session": self.session,
                "indices": list(self.indices), "sample_id": self.sample_id,
                "label": self.label, "path_score": self.path_score}


def load_city_development_data(config, protocol):
    windows, original, cache_identity = load_source_data(config, "city", severity_windows=None)
    manifest = read_json(Path(config["protocol_root"]) / "city/audit_manifest.json")
    fold = manifest["fold_manifest"]["folds"]["fold_1"]
    source_subjects = sorted(set(fold["train_subjects"]) | set(fold["val_subjects"]))
    outer_test = sorted(fold["test_subjects"])
    score_map = {}
    for subject in source_subjects:
        segments = manifest["audit"]["subjects"][subject]["segments"]
        for number, segment in enumerate(segments):
            if segment.get("path_score") is not None:
                score_map[f"{subject}/segment_{number:02d}"] = float(segment["path_score"])
    base_examples = sorted(original["severity"]["train"] + original["severity"]["val"],
                           key=lambda example: example.sample_id)
    if set(score_map) != {example.sample_id for example in base_examples}:
        raise RuntimeError("Continuous city path scores do not match source severity samples")
    examples = [ContinuousSeverityExample(
        example.subject_id, example.session, example.indices, example.label,
        example.sample_id, score_map[example.sample_id]
    ) for example in base_examples]
    partitions, split_audit = {}, {}
    for split_name, split in protocol["development_splits"].items():
        validation_subjects = set(split["val_subjects"])
        training_subjects = set(source_subjects) - validation_subjects
        if validation_subjects & set(outer_test) or len(validation_subjects) != 5:
            raise RuntimeError("Invalid nested validation subjects")
        train = [example for example in examples if example.subject_id in training_subjects]
        val = [example for example in examples if example.subject_id in validation_subjects]
        if {example.label for example in train} != {0, 1} or {example.label for example in val} != {0, 1}:
            raise RuntimeError("Each nested split must contain both binary classes")
        score_mean = float(np.mean([example.path_score for example in train]))
        score_std = float(np.std([example.path_score for example in train]))
        if not np.isfinite(score_std) or score_std <= 0:
            raise RuntimeError("Continuous training scores have no usable variation")
        partitions[split_name] = {"train": train, "val": val,
                                  "score_mean": score_mean, "score_std": score_std}
        split_audit[split_name] = {
            "train_subjects": sorted(training_subjects),
            "val_subjects": sorted(validation_subjects),
            "train_sha256": sample_hash(train), "val_sha256": sample_hash(val),
            "score_mean": score_mean, "score_std": score_std,
        }
    identity = {
        "dataset": "city", "source_subjects": source_subjects,
        "outer_test_subjects_excluded": outer_test,
        "development_splits": split_audit,
        "continuous_score_source": "city_audit_manifest_path_score",
        "all_examples_sha256": sample_hash(examples),
        "cache_identity_sha256": digest_json(cache_identity),
        "reference_calibration": False,
        "offline_transductive_subject_EA": True,
    }
    return windows, partitions, identity, cache_identity
