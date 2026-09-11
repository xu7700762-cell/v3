"""Nested source-only VRQ splits for paired state and severity development."""
from __future__ import annotations

from .token_data import load_source_data, sample_hash


def load_vrq_development_data(config, protocol):
    windows, original, cache_identity = load_source_data(config, "vrq", severity_windows=11)
    outer_source = set(cache_identity["split"]["train_subjects"]) | set(
        cache_identity["split"]["val_subjects"]
    )
    outer_test = set(cache_identity["split"]["test_subjects"])
    declared = [set(value["val_subjects"]) for value in protocol["development_splits"].values()]
    if set().union(*declared) != outer_source or sum(map(len, declared)) != len(outer_source):
        raise RuntimeError("Nested VRQ validation groups must partition all outer-source subjects")
    if outer_source & outer_test:
        raise RuntimeError("VRQ outer source and test identities overlap")
    partitions = {task: {} for task in ("severity", "state")}
    identity = {"outer_source_subjects": sorted(outer_source),
                "outer_test_subjects_excluded": sorted(outer_test), "tasks": {}}
    for task in partitions:
        examples = sorted(original[task]["train"] + original[task]["val"], key=lambda x: x.sample_id)
        if {example.subject_id for example in examples} != outer_source:
            raise RuntimeError(f"VRQ {task} source examples do not cover the outer-source identities")
        identity["tasks"][task] = {}
        for split, definition in protocol["development_splits"].items():
            val_subjects = set(definition["val_subjects"])
            train = [example for example in examples if example.subject_id not in val_subjects]
            val = [example for example in examples if example.subject_id in val_subjects]
            if {example.subject_id for example in train} & {example.subject_id for example in val}:
                raise RuntimeError("Nested VRQ identities overlap")
            if {example.label for example in train} != {0, 1} or {example.label for example in val} != {0, 1}:
                raise RuntimeError(f"Nested VRQ {task} split must contain both classes")
            partitions[task][split] = {"train": train, "val": val}
            identity["tasks"][task][split] = {
                "train_subjects": sorted({example.subject_id for example in train}),
                "val_subjects": sorted(val_subjects),
                "train_samples": len(train), "val_samples": len(val),
                "train_sha256": sample_hash(train), "val_sha256": sample_hash(val),
                "train_positive": sum(example.label for example in train),
                "val_positive": sum(example.label for example in val),
            }
    return windows, partitions, identity, cache_identity
