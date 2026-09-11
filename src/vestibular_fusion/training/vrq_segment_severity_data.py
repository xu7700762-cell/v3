"""Weakly supervised VRQ severity windows with subject-disjoint nested splits."""
from __future__ import annotations

from .token_data import TokenExample, load_source_data, sample_hash


def load_vrq_segment_severity_data(config, protocol):
    windows, original, cache_identity = load_source_data(config, "vrq", severity_windows=11)
    severity = sorted(original["severity"]["train"] + original["severity"]["val"],
                      key=lambda example: example.sample_id)
    state = sorted(original["state"]["train"] + original["state"]["val"],
                   key=lambda example: example.sample_id)
    targets = {(example.subject_id, example.session): example.label for example in severity}
    examples = []
    for example in state:
        key = (example.subject_id, example.session)
        if key not in targets:
            continue
        examples.append(TokenExample(
            example.subject_id, example.session, example.indices, targets[key],
            f"{example.subject_id}/{example.session}/severity_{example.sample_id.rsplit('/', 1)[-1]}",
        ))
    outer_source = set(cache_identity["split"]["train_subjects"]) | set(
        cache_identity["split"]["val_subjects"]
    )
    outer_test = set(cache_identity["split"]["test_subjects"])
    if {example.subject_id for example in examples} != outer_source or outer_source & outer_test:
        raise RuntimeError("Segment-level VRQ source identities are incomplete or overlap outer-test")
    declared = [set(value["val_subjects"]) for value in protocol["development_splits"].values()]
    if set().union(*declared) != outer_source or sum(map(len, declared)) != len(outer_source):
        raise RuntimeError("Nested segment validation groups must partition all outer-source subjects")
    partitions, split_identity = {}, {}
    for split, definition in protocol["development_splits"].items():
        val_subjects = set(definition["val_subjects"])
        train = [example for example in examples if example.subject_id not in val_subjects]
        val = [example for example in examples if example.subject_id in val_subjects]
        if {example.subject_id for example in train} & {example.subject_id for example in val}:
            raise RuntimeError("A VRQ subject appears in both segment train and validation")
        if {example.label for example in train} != {0, 1} or {example.label for example in val} != {0, 1}:
            raise RuntimeError("Segment-level train and validation must contain both classes")
        partitions[split] = {"train": train, "val": val}
        split_identity[split] = {
            "train_subjects": sorted({example.subject_id for example in train}),
            "val_subjects": sorted(val_subjects),
            "train_segments": len(train), "val_segments": len(val),
            "train_sha256": sample_hash(train), "val_sha256": sample_hash(val),
            "train_subject_segment_counts": {
                subject: sum(example.subject_id == subject for example in train)
                for subject in sorted({example.subject_id for example in train})},
            "val_subject_segment_counts": {
                subject: sum(example.subject_id == subject for example in val)
                for subject in sorted(val_subjects)},
        }
    identity = {
        "target": "subject_final_ssq_binary_label_broadcast_to_final_task_windows",
        "evaluation_unit": "five_second_final_task_window",
        "weak_supervision": True,
        "outer_source_subjects": sorted(outer_source),
        "outer_test_subjects_excluded": sorted(outer_test),
        "splits": split_identity,
    }
    return windows, partitions, identity, cache_identity
