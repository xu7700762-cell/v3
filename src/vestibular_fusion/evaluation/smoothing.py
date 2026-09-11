from __future__ import annotations

import numpy as np


def smooth_current_future(rows: list[dict], values: np.ndarray, windows: int = 3) -> np.ndarray:
    """Apply [t,t+1,t+2] within one subject/session, replicating the right edge."""
    values = np.asarray(values)
    if len(rows) != len(values) or int(windows) <= 0:
        raise ValueError("Smoothing expects aligned rows/values and a positive window count")
    output = np.empty_like(values, dtype=np.float64)
    groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for row_index, row in enumerate(rows):
        groups.setdefault((str(row["subject_id"]), str(row["session"])), []).append(
            (int(row["window_index"]), row_index)
        )
    for group in groups.values():
        indices = np.asarray([index for _, index in sorted(group)], dtype=np.int64)
        current = values[indices]
        for local_index, row_index in enumerate(indices):
            positions = [min(local_index + offset, len(indices) - 1) for offset in range(windows)]
            output[row_index] = current[positions].mean(axis=0)
    return output
