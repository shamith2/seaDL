from typing import Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

import mlx.core as mx

from .base import Module
from .activation import ReLU
from .linear import Linear, Flatten


@jaxtyped(typechecker=typechecker)
class SimpleMLP(Module):
    def __init__(
            self,
            in_features: int = 28*28,
            out_features: int = 10,
            hidden_features: int = 100,
            device: Optional[mx.DeviceType] = None
    ):
        """
        Simple Multi-layer Perceptron (MLP) with 1 hidden layer
        """
        super().__init__()

        self.device = mx.gpu if not device else device

        self.flatten_layer = Flatten(start_dim=1, end_dim=-1, device=self.device)
        self.linear1_layer = Linear(in_features=in_features, out_features=hidden_features, bias=True, device=self.device)
        self.linear2_layer = Linear(in_features=hidden_features, out_features=out_features, bias=True, device=self.device)

        self.relu1 = ReLU()

    def __call__(
            self,
            x: mx.array
    ) -> mx.array:
        x_flatten = self.flatten_layer(x)

        out1 = self.relu1(self.linear1_layer(x_flatten))
        out = self.linear2_layer(out1)

        return out

