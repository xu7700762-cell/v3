from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from ..data.anchors import anchor_indices, task_indices
from ..ssl_protocol import digest_json


@dataclass(frozen=True)
class ProbeExample:
    subject_id: str
    session: str
    indices: tuple[int, ...]
    label: int
    sample_id: str

    def identity(self) -> dict:
        return {"sample_id": self.sample_id, "subject_id": self.subject_id,
                "session": self.session, "window_indices": list(self.indices)}


def make_examples(bank, fold, task: str, subjects) -> list[ProbeExample]:
    subjects = set(subjects)
    available = set(fold.source_subjects) | set(fold.test_subjects)
    if not subjects or not subjects <= available or not subjects <= set(bank.records):
        raise ValueError("Invalid or empty subject partition")
    severity = [e for e in (*fold.source_examples, *fold.test_examples)
                if e.subject_id in subjects]
    if task == "severity":
        result = [ProbeExample(e.subject_id, e.task_session, tuple(int(i) for i in task_indices(
            bank.records[e.subject_id], e.task_session, 11
        )), int(e.label), f"{e.subject_id}/{e.task_session}") for e in severity]
    elif task == "state":
        references = {}
        for e in severity:
            previous = references.setdefault(e.subject_id, e.reference_session)
            if previous != e.reference_session:
                raise ValueError("Inconsistent reference session")
        if set(references) != subjects:
            raise ValueError("Missing reference session for state anchor exclusion")
        excluded = {s: set(int(i) for i in anchor_indices(bank.records[s], references[s]))
                    for s in subjects}
        result = [ProbeExample(s, str(record.sessions[i]), (i,), int(record.labels[i]),
                               f"{s}/{i}")
                  for s in sorted(subjects) for record in [bank.records[s]]
                  for i in range(len(record.windows)) if i not in excluded[s]]
    else:
        raise ValueError(f"Unknown task: {task}")
    result.sort(key=lambda e: e.sample_id)
    if not result or len({e.sample_id for e in result}) != len(result):
        raise ValueError("Empty or duplicate probe examples")
    if {e.subject_id for e in result} != subjects:
        raise ValueError("Every selected subject must contribute examples")
    return result


def input_tensor(bank, examples: list[ProbeExample], task: str, device) -> torch.Tensor:
    values = [bank.records[e.subject_id].windows[list(e.indices)] for e in examples]
    array = np.stack(values).astype(np.float32)
    if task == "state":
        array = array[:, 0]
    return torch.as_tensor(array, device=device)


def class_weight(examples: list[ProbeExample]) -> float:
    labels = [e.label for e in examples]
    if set(labels) != {0, 1}:
        raise ValueError("Training partition must contain both binary classes")
    return labels.count(0) / labels.count(1)


def epoch_order(count: int, seed: int, epoch: int) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), 2])).permutation(count)


def sample_digest(examples: list[ProbeExample]) -> str:
    return digest_json([e.identity() for e in examples])
