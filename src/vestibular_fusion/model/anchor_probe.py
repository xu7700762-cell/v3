from contextlib import nullcontext

import torch

from .linear_probe import FEMBALinearProbe, build_probe, tensor_state_hash


class FEMBAAnchorProbe(FEMBALinearProbe):
    """A paired linear probe with optional, differentiable subject centering."""

    def __init__(self, encoder, *, encoder_trainable, seed, condition):
        if condition not in ("C0", "C1"):
            raise ValueError("Unknown anchor condition")
        super().__init__(encoder, encoder_trainable=encoder_trainable, seed=seed)
        self.condition = condition
        self.branch_gradients = {"target": False, "anchor": False}

    def _pooled(self, windows, branch):
        tokens = self.encoder.forward_tokens(windows.reshape(-1, 30, 1280).float())
        if tokens.shape[1:] != (80, 525):
            raise ValueError("Unexpected FEMBA token shape")
        values = tokens.float().mean(1)
        if self.training and values.requires_grad:
            def audit(grad):
                self.branch_gradients[branch] |= bool(torch.any(grad != 0))
            values.register_hook(audit)
        return values

    def forward(self, windows, anchors=None, subject_index=None):
        if windows.ndim not in (3, 4) or windows.shape[-2:] != (30, 1280):
            raise ValueError("Expected target windows [B,30,1280] or [B,11,30,1280]")
        if windows.ndim == 4 and windows.shape[1] != 11:
            raise ValueError("Severity requires 11 windows")
        with nullcontext() if self.encoder_trainable else torch.no_grad():
            z = self._pooled(windows, "target")
            if windows.ndim == 4:
                z = z.reshape(len(windows), 11, 525).mean(1)
            if self.condition == "C1":
                if anchors is None or anchors.ndim != 4 or anchors.shape[1:] != (4, 30, 1280):
                    raise ValueError("C1 requires four anchors per unique subject")
                if subject_index is None or subject_index.shape != (len(windows),):
                    raise ValueError("Missing target-to-anchor subject mapping")
                center = self._pooled(anchors, "anchor").reshape(len(anchors), 4, 525).mean(1)
                z = z - center[subject_index]
            elif anchors is not None or subject_index is not None:
                raise ValueError("C0 must not encode anchors")
        return self.head(z).squeeze(-1)


def build_anchor_probe(variant, seed, device, *, condition, model_factory=None, **kwargs):
    base, initial = build_probe(variant, seed, torch.device("cpu"), **kwargs)
    factory = FEMBAAnchorProbe if model_factory is None else model_factory
    model = factory(base.encoder, encoder_trainable=base.encoder_trainable,
                            seed=seed, condition=condition)
    if model_factory is None:
        model.head.load_state_dict(base.head.state_dict())
    else:
        initial.update(head_sha256=tensor_state_hash(model.head.state_dict()),
                       head_parameters=sum(p.numel() for p in model.head.parameters()),
                       parameter_shapes={n: list(p.shape) for n, p in model.named_parameters()})
        initial.update(model.head_initialization())
    return model.to(device), initial
