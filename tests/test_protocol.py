from pathlib import Path

from vestibular_fusion.cli import build_parser
from vestibular_fusion.protocol import PROTOCOL


def test_protocol_locks_the_simplified_multitask_model():
    assert PROTOCOL["training_seed"] == 2001
    assert PROTOCOL["encoder"] == "pretrained_femba_four_block_frozen"
    assert PROTOCOL["training_input"].startswith("sha256_bound_cached")
    assert PROTOCOL["evaluation_input"].startswith("raw_windows")
    assert PROTOCOL["shared_head"].startswith("FractionalDoGPolynomialKAN")
    assert PROTOCOL["anchor_count"] == 4
    assert PROTOCOL["severity_task_windows"] == 11
    assert PROTOCOL["severity_weight"] == 0.5
    assert PROTOCOL["state_deviation_learning"].startswith("subject-wise ST-SSDL-inspired")
    assert PROTOCOL["state_deviation_weight"] == 0.3
    assert PROTOCOL["state_deviation_margin"] == 0.25
    assert PROTOCOL["severity_head_lr"] == 1e-3
    assert PROTOCOL["severity_task_sampling"].startswith("train_jittered_uniform")
    assert PROTOCOL["severity_gradient_scope"] == "shared_kan_and_severity_head"
    assert PROTOCOL["selection_key"].startswith("maximum_validation_joint_balanced_accuracy")
    assert PROTOCOL["state_decision"] == "anchor_oriented_subject_kmeans_160d"
    assert PROTOCOL["severity_threshold"] == 0.5
    assert PROTOCOL["severity_bias_calibration"].startswith("source_balanced_accuracy")
    assert PROTOCOL["severity_residual_bound"] == 0.1
    assert PROTOCOL["severity_base_normalization"].startswith("source_only_frozen_femba")
    assert PROTOCOL["selection_min_epochs"] == 20
    assert PROTOCOL["selection_max_epochs"] == 60
    assert PROTOCOL["selection_patience"] == 15
    assert PROTOCOL["state_smoothing"].startswith("secondary_only")


def test_cli_only_exposes_current_commands():
    parser = build_parser()
    subparsers = next(action for action in parser._actions if isinstance(action.choices, dict))
    assert set(subparsers.choices) == {"preflight", "train", "evaluate"}


def test_train_cli_accepts_an_explicit_training_seed():
    args = build_parser().parse_args(
        [
            "train",
            "--config",
            "paths.json",
            "--dataset",
            "vrq",
            "--training-seed",
            "2001",
        ]
    )
    assert args.training_seed == 2001


def test_train_cli_defaults_to_v27_seed_2001():
    args = build_parser().parse_args(
        ["train", "--config", "paths.json", "--dataset", "vrq"]
    )
    assert args.training_seed == 2001
    assert args.projection_variant == "fractional_dog_polykan"


def test_removed_downstream_symbols_are_absent_from_source():
    root = Path(__file__).resolve().parents[1] / "src" / "vestibular_fusion"
    forbidden = (
        "DirectionalMambaKAN",
        "PairSeverityHead",
        "PolynomialKANLayer(96",
        "lorentz_distance",
        "fixed_feature_bank",
        "fit_source_head",
        "R1 + R2",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    assert not any(value in text for value in forbidden)
