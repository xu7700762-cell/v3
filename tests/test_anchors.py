import numpy as np
import pytest

from vestibular_fusion.data.anchors import (
    anchor_indices,
    context_indices,
    context_windows,
    jittered_task_indices,
    task_indices,
)
from vestibular_fusion.data.types import SubjectRecord


def _record(reference: int = 16, task: int = 11) -> SubjectRecord:
    return SubjectRecord(
        windows=np.zeros((reference + task, 30, 1280), dtype=np.float16),
        labels=np.asarray([0] * reference + [1] * task, dtype=np.int64),
        sessions=["rest"] * reference + ["task"] * task,
    )


def test_u3_u6_anchor_selection_is_fixed_and_unique():
    assert anchor_indices(_record(), "rest").tolist() == [4, 6, 9, 11]
    assert task_indices(_record(), "task", 11).tolist() == list(range(16, 27))


def test_missing_anchor_or_task_windows_fail_visibly():
    with pytest.raises(ValueError, match="at least 8"):
        anchor_indices(_record(reference=7), "rest")
    with pytest.raises(ValueError, match="at least 11"):
        task_indices(_record(task=10), "task", 11)


def test_context_stays_inside_session_and_replicates_right_edge():
    record = _record()
    assert context_indices(record, 15).tolist() == [15, 15, 15]
    assert context_indices(record, 16).tolist() == [16, 17, 18]
    assert context_windows(record, np.asarray([15, 16])).shape == (2, 3, 30, 1280)


def test_jittered_task_selection_is_reproducible_unique_and_near_uniform():
    record = _record(task=60)
    first = jittered_task_indices(record, "task", 11, seed=7)
    second = jittered_task_indices(record, "task", 11, seed=7)
    different = jittered_task_indices(record, "task", 11, seed=8)
    locked = task_indices(record, "task", 11)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 11
    assert np.all(np.abs(first - locked) <= 2)
    assert not np.array_equal(first, different)
