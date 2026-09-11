import numpy as np
import torch

from vestibular_fusion.model.vrq_boosted import VRQBoostedHead
from vestibular_fusion.training.token_data import TokenExample
from vestibular_fusion.training.vrq_segment_severity import score_segment_severity


def test_segment_scoring_keeps_window_and_subject_units_separate():
    tokens = np.random.default_rng(9).normal(size=(4, 80, 525)).astype(np.float32)
    examples = [TokenExample("low", "task", (0,), 0, "low/0"),
                TokenExample("low", "task", (1,), 0, "low/1"),
                TokenExample("high", "task", (2,), 1, "high/0"),
                TokenExample("high", "task", (3,), 1, "high/1")]
    head = VRQBoostedHead("base", "state", 2001)
    rows, result = score_segment_severity(head, tokens, examples, torch.device("cpu"), 2)
    assert len(rows) == 4
    assert result["metrics"]["n_subjects"] == 2
    assert result["subject_audit"]["subject_count"] == 2
    assert 0 <= result["subject_audit"]["subject_macro_segment_accuracy"] <= 1
