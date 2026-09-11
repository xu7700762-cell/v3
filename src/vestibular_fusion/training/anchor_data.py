import numpy as np
import torch

from ..data.anchors import anchor_indices
from ..ssl_protocol import digest_json
from .probe_data import input_tensor, make_examples, sample_digest


def pilot_partitions(bank, fold, task):
    """Never construct outer-test target examples for this pilot."""
    train_ids, val_ids, test_ids = map(set, (fold.source_train_subjects,
                                          fold.source_val_subjects, fold.test_subjects))
    if train_ids & val_ids or (train_ids | val_ids) & test_ids:
        raise ValueError("Pilot identities overlap")
    if train_ids | val_ids != set(fold.source_subjects):
        raise ValueError("Invalid source partition")
    references = {}
    for e in fold.source_examples:
        if e.subject_id not in train_ids | val_ids:
            raise ValueError("Reference subject outside source")
        if references.setdefault(e.subject_id, e.reference_session) != e.reference_session:
            raise ValueError("Conflicting reference sessions")
    if set(references) != train_ids | val_ids:
        raise ValueError("Missing source reference session")
    anchors = {s: {"reference_session": ref,
                   "indices": [int(i) for i in anchor_indices(bank.records[s], ref)]}
               for s, ref in sorted(references.items())}
    train = make_examples(bank, fold, task, train_ids)
    val = make_examples(bank, fold, task, val_ids)
    identity = {"train_samples_sha256": sample_digest(train),
                "val_samples_sha256": sample_digest(val),
                "anchors": anchors, "anchors_sha256": digest_json(anchors)}
    return train, val, anchors, identity


def batch_inputs(bank, examples, task, device, condition, anchors):
    windows = input_tensor(bank, examples, task, device)
    encoded = len(examples) * (11 if task == "severity" else 1)
    if condition == "C0":
        return (windows, None, None), encoded
    subjects = sorted({e.subject_id for e in examples})
    for s in subjects:
        if s not in anchors:
            raise ValueError("No anchors for target subject")
        info = anchors[s]
        record = bank.records[s]
        if len(set(info["indices"])) != 4 or any(
                str(record.sessions[i]) != info["reference_session"] for i in info["indices"]):
            raise ValueError("Invalid subject reference anchors")
    reference = np.stack([bank.records[s].windows[anchors[s]["indices"]] for s in subjects])
    indices = torch.tensor([subjects.index(e.subject_id) for e in examples], device=device)
    return (windows, torch.as_tensor(reference.astype(np.float32), device=device), indices), encoded + 4 * len(subjects)
