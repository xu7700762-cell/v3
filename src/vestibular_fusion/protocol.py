from __future__ import annotations


DEFAULT_STATE_THRESHOLD = 0.5
STATE_DECISION = "anchor_oriented_subject_kmeans_160d"
SEVERITY_THRESHOLD = 0.5


PROTOCOL = {
    "name": "femba_kan_mtl_subject_fivefold",
    "split_seed": 42,
    "training_seed": 2001,
    "outer_folds": 5,
    "training_schedule": "source_validation_joint_metric_selection_then_source_refit",
    "encoder": "pretrained_femba_four_block_frozen",
    "pooling": "token_mean",
    "training_input": "sha256_bound_cached_frozen_femba_token_mean_525_plus_token_dynamics_2",
    "evaluation_input": "raw_windows_recomputed_by_frozen_femba",
    "shared_head": "FractionalDoGPolynomialKAN(525,160,degree=2) with anchor-relative state and token-dynamics PRISM severity",
    "anchor_calibration": "U3-U6 distributed 15-second context embedding center",
    "anchor_count": 4,
    "severity_task_windows": 11,
    "severity_weight": 0.5,
    "state_deviation_learning": "subject-wise ST-SSDL-inspired anchor deviation pairwise margin",
    "state_deviation_weight": 0.3,
    "state_deviation_margin": 0.25,
    "state_decision": STATE_DECISION,
    "severity_threshold": SEVERITY_THRESHOLD,
    "severity_bias_calibration": "source_balanced_accuracy_bias_shift_v1",
    "severity_residual_bound": 0.1,
    "severity_base_normalization": "source_only_frozen_femba_token_dynamics_zscore_v2",
    "selection_key": "maximum_validation_joint_balanced_accuracy_then_auroc_after_min_epochs",
    "selection_min_epochs": 20,
    "selection_max_epochs": 60,
    "selection_patience": 15,
    "severity_batching": "balanced_2_low_2_high",
    "severity_task_sampling": "train_jittered_uniform_eval_locked_uniform_11",
    "severity_gradient_scope": "shared_kan_and_severity_head",
    "head_lr": 2e-5,
    "severity_head_lr": 1e-3,
    "weight_decay": 1e-2,
    "head_dropout": 0.2,
    "component_initialization": "separate_seed_streams_v1",
    "label_smoothing": 0.05,
    "state_smoothing": "secondary_only_[t,t+1,t+2]",
}
