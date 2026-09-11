"""C0-only nonlinear heads, with a shared output initialization and no reference inputs."""
from functools import partial

import torch
from torch import nn

from .anchor_probe import FEMBAAnchorProbe
from .kan import FractionalDoGPolynomialKANLayer
from .main import ParameterMatchedMLP
from .linear_probe import tensor_state_hash

HEAD_PARAMETERS = {"fractional_dog_polykan": 170749, "mlp": 170447}


class FEMBAC0HeadProbe(FEMBAAnchorProbe):
    def __init__(self, encoder, *, encoder_trainable, seed, condition, head_kind):
        if condition != "C0" or head_kind not in HEAD_PARAMETERS:
            raise ValueError("Head comparison accepts C0 and the two locked nonlinear heads only")
        super().__init__(encoder, encoder_trainable=encoder_trainable, seed=seed, condition=condition)
        self.head_kind = head_kind
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 101)
            projection = (FractionalDoGPolynomialKANLayer(525, 160, degree=2)
                          if head_kind == "fractional_dog_polykan" else ParameterMatchedMLP(525, 160, 246))
            torch.manual_seed(int(seed) + 211)
            self.head = nn.Sequential(nn.LayerNorm(525), projection, nn.LayerNorm(160), nn.Linear(160, 1))
        if sum(p.numel() for p in self.head.parameters()) != HEAD_PARAMETERS[head_kind]:
            raise AssertionError("Locked downstream parameter count changed")

    def head_initialization(self):
        return {"head_kind": self.head_kind,
                "projection_sha256": tensor_state_hash(self.head[1].state_dict()),
                "output_layer_sha256": tensor_state_hash(self.head[3].state_dict()),
                "normalization_sha256": tensor_state_hash({
                    **{"input." + k: v for k, v in self.head[0].state_dict().items()},
                    **{"latent." + k: v for k, v in self.head[2].state_dict().items()}})}


def head_factory(head_kind):
    if head_kind not in HEAD_PARAMETERS:
        raise ValueError("Unknown nonlinear head")
    return partial(FEMBAC0HeadProbe, head_kind=head_kind)
