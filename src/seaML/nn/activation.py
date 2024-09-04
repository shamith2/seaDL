from typing import Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

import mlx.core as mx

from .base import Tensor, Module
from .functional import relu


@jaxtyped(typechecker=typechecker)
class ReLU(Module):
    def __init__(
            self,
            device: Optional[mx.DeviceType] = None
    ):
        super().__init__()

        self.device = device


    def __call__(
            self,
            x: Tensor
    ) -> Tensor:
        """
        Like torch.nn.ReLU using functional relu
        """
        # return relu(x, self.device)

        return relu(x)


    def extra_repr(self):
        return super().extra_repr()

