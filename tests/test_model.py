from __future__ import annotations

import torch
import torch.nn as nn

from vestibular_fusion.model.main import FEMBAKANMultiTaskModel, PRISMResidualSeverityHead
from vestibular_fusion.training.runner import (
    ADAPTIVE_BASIS_LR,
    HEAD_LR,
    SEVERITY_HEAD_LR,
    _optimizer,
    _state_deviation_loss,
)


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward_tokens(self, windows: torch.Tensor) -> torch.Tensor:
        base = windows.mean(dim=(1, 2), keepdim=False)[:, None, None]
        positions = torch.linspace(0.0, 1.0, 80, device=windows.device)[None, :, None]
        features = torch.linspace(-1.0, 1.0, 525, device=windows.device)[None, None, :]
        return self.scale * (base + positions + features)

    def model_spec(self) -> dict:
        return {"name": "fake", "token_shape": [80, 525]}


def _model() -> FEMBAKANMultiTaskModel:
    return FEMBAKANMultiTaskModel(encoder=FakeEncoder(), seed=7)


def test_mlp_projection_is_parameter_matched_to_kan():
    kan = FEMBAKANMultiTaskModel(
        encoder=FakeEncoder(), projection_variant="kan", seed=7
    )
    mlp = FEMBAKANMultiTaskModel(
        encoder=FakeEncoder(), projection_variant="mlp", seed=7
    )
    kan_parameters = sum(parameter.numel() for parameter in kan.shared_kan.parameters())
    mlp_parameters = sum(parameter.numel() for parameter in mlp.shared_kan.parameters())
    assert kan_parameters == 168160
    assert mlp_parameters == 168230
    assert (mlp_parameters - kan_parameters) / kan_parameters < 0.001
    assert mlp.encode_pooled(torch.randn(3, 525)).shape == (3, 160)
    assert mlp.model_spec()["projection_variant"] == "mlp"


def test_projection_variants_share_identical_task_head_initialization():
    models = [
        FEMBAKANMultiTaskModel(
            encoder=FakeEncoder(),
            kan_degree=2,
            projection_variant=variant,
            seed=17,
        )
        for variant in ("kan", "mlp", "fractional_dog_polykan")
    ]
    for name in models[0].state_head.state_dict():
        assert all(
            torch.equal(
                models[0].state_head.state_dict()[name],
                model.state_head.state_dict()[name],
            )
            for model in models[1:]
        )
    for name in models[0].severity_head.state_dict():
        assert all(
            torch.equal(
                models[0].severity_head.state_dict()[name],
                model.severity_head.state_dict()[name],
            )
            for model in models[1:]
        )


def test_fractional_dog_polykan_is_parameter_matched_and_stable():
    model = FEMBAKANMultiTaskModel(
        encoder=FakeEncoder(),
        kan_degree=2,
        projection_variant="fractional_dog_polykan",
        seed=7,
    )
    mlp = FEMBAKANMultiTaskModel(
        encoder=FakeEncoder(), projection_variant="mlp", seed=7
    )
    model_parameters = sum(
        parameter.numel() for parameter in model.shared_kan.parameters()
    )
    mlp_parameters = sum(parameter.numel() for parameter in mlp.shared_kan.parameters())
    assert model_parameters == 169218
    assert abs(model_parameters - mlp_parameters) / mlp_parameters < 0.01
    values = model.encode_pooled(torch.randn(4, 525) * 100.0)
    assert values.shape == (4, 160)
    assert torch.isfinite(values).all()
    assert model.model_spec()["kan_degree"] == 2
    assert model.model_spec()["projection_variant"] == "fractional_dog_polykan"


def test_fractional_dog_polykan_starts_from_the_degree_two_kan_mapping():
    kan = FEMBAKANMultiTaskModel(
        encoder=FakeEncoder(), projection_variant="kan", seed=7
    ).eval()
    pooled = torch.randn(32, 525)
    with torch.no_grad():
        kan_values = kan.encode_pooled(pooled)
        residual = FEMBAKANMultiTaskModel(
            encoder=FakeEncoder(),
            kan_degree=2,
            projection_variant="fractional_dog_polykan",
            seed=7,
        ).eval()
        residual_values = residual.encode_pooled(pooled)
        assert torch.allclose(kan_values, residual_values, atol=1e-3, rtol=1e-3)


def test_fractional_dog_basis_is_identity_initialized_and_learnable():
    model = FEMBAKANMultiTaskModel(
        encoder=FakeEncoder(),
        kan_degree=2,
        projection_variant="fractional_dog_polykan",
        seed=7,
    )
    values = torch.randn(8, 525)
    _, _, fractional, dog = model.shared_kan.basis_features(values)
    assert torch.allclose(fractional, torch.tanh(values), atol=1e-6, rtol=1e-6)
    assert dog.shape == values.shape
    assert torch.isfinite(dog).all()
    model.shared_kan(values).square().mean().backward()
    assert model.shared_kan.fractional_order_logit.grad.abs().item() > 0.0
    assert model.shared_kan.dog_mix_logits.grad.abs().sum().item() > 0.0


def test_fractional_dog_polykan_uses_a_separate_basis_learning_rate():
    model = FEMBAKANMultiTaskModel(
        encoder=FakeEncoder(),
        kan_degree=2,
        projection_variant="fractional_dog_polykan",
        seed=7,
    )
    optimizer = _optimizer(model)
    assert [group["lr"] for group in optimizer.param_groups] == [
        HEAD_LR,
        ADAPTIVE_BASIS_LR,
        SEVERITY_HEAD_LR,
    ]
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(
        id(parameter) in optimized for parameter in model.shared_kan.parameters()
    )
    model.adaptive_basis_lr = 1e-4
    optimizer = _optimizer(model)
    assert [group["lr"] for group in optimizer.param_groups] == [
        HEAD_LR,
        1e-4,
        SEVERITY_HEAD_LR,
    ]


def test_model_shapes_and_anchor_centering():
    model = _model()
    windows = torch.randn(2, 6, 30, 1280)
    anchors = torch.randn(2, 4, 30, 1280)
    encoded = model.encode_windows(windows.reshape(-1, 30, 1280))
    assert encoded.shape == (12, 160)
    center = model.anchor_center(anchors)
    assert center.shape == (2, 160)
    calibrated_anchors = model.calibrated_embeddings(anchors, anchors)
    assert torch.allclose(calibrated_anchors.mean(dim=1), torch.zeros_like(center), atol=1e-5)
    assert model.state_logits(windows, anchors).shape == (2, 6)
    assert model.state_logits(windows[0], anchors[0]).shape == (6,)
    assert model.severity_logits(windows, anchors).shape == (2,)


def test_context_shapes_use_embedding_mean_without_changing_the_heads():
    model = _model()
    contexts = torch.randn(7, 3, 30, 1280)
    anchors = torch.randn(2, 4, 3, 30, 1280)
    windows = torch.randn(2, 6, 3, 30, 1280)
    assert model.encode_contexts(contexts).shape == (7, 160)
    assert model.anchor_center(anchors).shape == (2, 160)
    assert model.state_logits(windows, anchors).shape == (2, 6)
    assert model.severity_logits(windows, anchors).shape == (2,)
    assert isinstance(model.severity_head, PRISMResidualSeverityHead)


def test_cached_pooled_features_match_the_raw_frozen_encoder_path():
    model = _model().eval()
    windows = torch.randn(2, 6, 3, 30, 1280)
    anchors = torch.randn(2, 4, 3, 30, 1280)
    pooled_windows, token_dynamics = model.extract_frozen_features(
        windows.reshape(-1, 30, 1280)
    )
    pooled_windows = pooled_windows.reshape(2, 6, 3, 525)
    token_dynamics = token_dynamics.reshape(2, 6, 3, 2)
    pooled_anchors = model.pool_windows(anchors.reshape(-1, 30, 1280)).reshape(
        2, 4, 3, 525
    )
    with torch.no_grad():
        assert torch.equal(
            model.state_logits(windows, anchors),
            model.state_logits_from_pooled(pooled_windows, pooled_anchors),
        )
        assert torch.equal(
            model.severity_logits(windows, anchors),
            model.severity_logits_from_pooled(
                pooled_windows, pooled_anchors, token_dynamics
            ),
        )


def test_severity_head_keeps_stable_base_and_direct_kan_features():
    model = _model()
    constant_pooled = torch.ones(2, 11, 525)
    constant_dynamics = torch.ones(2, 11, 2)
    constant = torch.ones(2, 11, 160)
    constant_features = model.severity_features(
        constant_pooled, constant_dynamics, constant, torch.ones(2)
    )
    assert constant_features.shape == (2, 163)
    assert torch.equal(constant_features[:, 0], torch.zeros(2))
    assert torch.equal(constant_features[:, 1], torch.ones(2))
    assert torch.equal(constant_features[:, 2], torch.zeros(2))
    assert torch.equal(constant_features[:, 3:], torch.ones(2, 160))

    increasing_pooled = torch.arange(11, dtype=torch.float32)[None, :, None].expand(
        2, 11, 525
    )
    increasing_dynamics = torch.arange(22, dtype=torch.float32).reshape(1, 11, 2).expand(
        2, 11, 2
    )
    increasing = torch.arange(11, dtype=torch.float32)[None, :, None].expand(2, 11, 160)
    increasing_features = model.severity_features(
        increasing_pooled, increasing_dynamics, increasing, torch.ones(2)
    )
    assert torch.all(increasing_features[:, 0] > 0.0)
    assert torch.all(increasing_features[:, 2] > 0.0)
    assert torch.all(increasing_features[:, 3:] > 0.0)


def test_prism_residual_cannot_overwrite_the_stable_dispersion_branch():
    model = _model()
    task_pooled = torch.randn(3, 11, 525)
    task_dynamics = torch.randn(3, 11, 2)
    calibrated = torch.randn(3, 11, 160)
    scale = torch.ones(3)
    with torch.no_grad():
        model.severity_head.correction_head.weight.fill_(100.0)
        base = model.severity_head.base_head(
            model.severity_head.normalized_base_feature(task_pooled, task_dynamics)
        ).squeeze(-1)
        full = model.severity_head(task_pooled, task_dynamics, calibrated, scale)
    residual = full - base - model.severity_head.output_bias
    assert torch.all(residual.abs() <= model.severity_head.residual_bound + 1e-7)


def test_severity_base_normalization_is_saved_in_the_head():
    model = _model()
    model.severity_head.set_base_normalization(
        [2.5, 1.5, -0.5], [0.75, 0.25, 0.5]
    )
    assert torch.equal(
        model.severity_head.base_center, torch.tensor([2.5, 1.5, -0.5])
    )
    assert torch.equal(
        model.severity_head.base_scale, torch.tensor([0.75, 0.25, 0.5])
    )


def test_attached_severity_training_updates_shared_kan_but_not_encoder():
    model = _model().train()
    windows = torch.randn(2, 11, 30, 1280)
    anchors = torch.randn(2, 4, 30, 1280)
    model.severity_logits(windows, anchors).sum().backward()
    assert all(parameter.grad is not None for parameter in model.shared_kan.parameters())
    assert all(parameter.grad is None for parameter in model.encoder.parameters())


def test_state_deviation_loss_rewards_anchor_distance_ordering():
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    ordered = _state_deviation_loss(torch.tensor([0.8, 1.0, 2.0, 2.2]), labels)
    reversed_order = _state_deviation_loss(
        torch.tensor([2.0, 2.2, 0.8, 1.0]), labels
    )
    assert ordered < reversed_order


def test_state_deviation_loss_compares_windows_within_each_subject():
    labels = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    separated_subject_scales = torch.tensor([[0.0, 1.0], [100.0, 101.0]])
    expected = torch.nn.functional.softplus(torch.tensor(-0.75))
    assert torch.allclose(
        _state_deviation_loss(separated_subject_scales, labels), expected
    )


def test_detached_severity_training_does_not_update_shared_kan():
    model = _model().train()
    windows = torch.randn(2, 11, 30, 1280)
    anchors = torch.randn(2, 4, 30, 1280)
    model.severity_logits(windows, anchors, detach_shared=True).sum().backward()
    assert all(parameter.grad is None for parameter in model.shared_kan.parameters())
    assert all(parameter.grad is not None for parameter in model.severity_head.parameters())


def test_encoder_stays_frozen_in_train_mode_and_optimizer_excludes_it():
    model = _model()
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    model.train()
    assert not model.encoder.training
    optimizer = _optimizer(model)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert [group["lr"] for group in optimizer.param_groups] == [
        HEAD_LR,
        ADAPTIVE_BASIS_LR,
        SEVERITY_HEAD_LR,
    ]
    assert all(id(parameter) not in optimized for parameter in model.encoder.parameters())
    assert all(id(parameter) in optimized for parameter in model.shared_kan.parameters())
    assert all(id(parameter) in optimized for parameter in model.state_head.parameters())
    assert all(id(parameter) in optimized for parameter in model.severity_head.parameters())


def test_checkpoint_round_trip_preserves_logits(tmp_path):
    first = _model().eval()
    second = _model().eval()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": first.state_dict()}, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    second.load_state_dict(payload["model_state_dict"], strict=True)
    windows = torch.randn(2, 5, 30, 1280)
    anchors = torch.randn(2, 4, 30, 1280)
    with torch.no_grad():
        assert torch.equal(first.state_logits(windows, anchors), second.state_logits(windows, anchors))
        assert torch.equal(
            first.severity_logits(windows, anchors), second.severity_logits(windows, anchors)
        )
