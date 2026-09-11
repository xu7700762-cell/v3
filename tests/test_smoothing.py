import numpy as np

from vestibular_fusion.evaluation.smoothing import smooth_current_future


def test_forward_smoothing_replicates_right_boundary_without_crossing_session():
    rows = [
        {"subject_id": "s1", "session": "a", "window_index": 0},
        {"subject_id": "s1", "session": "a", "window_index": 1},
        {"subject_id": "s1", "session": "b", "window_index": 0},
        {"subject_id": "s2", "session": "a", "window_index": 0},
    ]
    values = np.asarray([[1.0], [3.0], [100.0], [1000.0]])
    actual = smooth_current_future(rows, values, 3).ravel()
    np.testing.assert_allclose(actual, [7.0 / 3.0, 3.0, 100.0, 1000.0])


def test_forward_smoothing_uses_window_order_not_input_order():
    rows = [
        {"subject_id": "s1", "session": "a", "window_index": 1},
        {"subject_id": "s1", "session": "a", "window_index": 0},
    ]
    values = np.asarray([[3.0], [1.0]])
    actual = smooth_current_future(rows, values, 3).ravel()
    np.testing.assert_allclose(actual, [3.0, 7.0 / 3.0])
