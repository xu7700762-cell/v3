from __future__ import annotations

import numpy as np

from .types import SubjectRecord


CONTEXT_WINDOWS = 3


def uniform_indices(indices: list[int], count: int) -> np.ndarray:
    if len(indices) < int(count):
        raise ValueError(f"Need at least {count} windows, found {len(indices)}")
    positions = np.rint(np.linspace(0, len(indices) - 1, int(count))).astype(np.int64)
    if len(np.unique(positions)) != int(count):
        raise ValueError(f"Cannot select {count} unique uniform windows from {len(indices)}")
    return np.asarray([indices[int(position)] for position in positions], dtype=np.int64)


def jittered_uniform_indices(
    indices: list[int], count: int, seed: int, max_jitter: int = 2
) -> np.ndarray:
    if len(indices) < int(count):
        raise ValueError(f"Need at least {count} windows, found {len(indices)}")
    base = np.rint(np.linspace(0, len(indices) - 1, int(count))).astype(np.int64)
    offsets = np.random.RandomState(int(seed)).randint(
        -int(max_jitter), int(max_jitter) + 1, size=int(count)
    )
    positions = []
    for slot, (position, offset) in enumerate(zip(base, offsets)):
        lower = positions[-1] + 1 if positions else 0
        upper = len(indices) - (int(count) - slot)
        positions.append(int(np.clip(position + offset, lower, upper)))
    if len(np.unique(positions)) != int(count):
        raise AssertionError("Jittered uniform selection must remain unique")
    return np.asarray([indices[position] for position in positions], dtype=np.int64)


def session_indices(record: SubjectRecord, session: str) -> list[int]:
    return [index for index, value in enumerate(record.sessions) if str(value) == str(session)]


def anchor_indices(record: SubjectRecord, reference_session: str) -> np.ndarray:
    slots = uniform_indices(session_indices(record, reference_session), 8)
    selected = slots[np.asarray([2, 3, 4, 5], dtype=np.int64)]
    if len(np.unique(selected)) != 4:
        raise AssertionError("U3-U6 must select exactly four unique anchors")
    return selected


def task_indices(record: SubjectRecord, task_session: str, count: int = 11) -> np.ndarray:
    return uniform_indices(session_indices(record, task_session), int(count))


def jittered_task_indices(
    record: SubjectRecord, task_session: str, count: int, seed: int
) -> np.ndarray:
    return jittered_uniform_indices(
        session_indices(record, task_session), int(count), int(seed)
    )


def context_indices(
    record: SubjectRecord, index: int, count: int = CONTEXT_WINDOWS
) -> np.ndarray:
    index = int(index)
    session = str(record.sessions[index])
    indices = session_indices(record, session)
    position = indices.index(index)
    return np.asarray(
        [indices[min(position + offset, len(indices) - 1)] for offset in range(int(count))],
        dtype=np.int64,
    )


def context_windows(
    record: SubjectRecord,
    indices: list[int] | np.ndarray,
    count: int = CONTEXT_WINDOWS,
) -> np.ndarray:
    return np.stack(
        [record.windows[context_indices(record, int(index), count)] for index in indices]
    )


def context_features(
    record: SubjectRecord,
    features: np.ndarray,
    indices: list[int] | np.ndarray,
    count: int = CONTEXT_WINDOWS,
) -> np.ndarray:
    if features.ndim != 2 or features.shape[0] != len(record.windows):
        raise ValueError(
            "Cached features must align one-to-one with the subject raw windows"
        )
    return np.stack(
        [features[context_indices(record, int(index), count)] for index in indices]
    )
