from typing import Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from ..base import Tensor, Device
from .base import Module
from .functional import relu


@jaxtyped(typechecker=typechecker)
class ReLU(Module):
    def __init__(
            self,
            device: Optional[Device] = None
    ):
        super().__init__()


    def __call__(
            self,
            x: Tensor
    ) -> Tensor:
        """
        Like torch.nn.ReLU using functional relu
        """
        return relu(x)


    def extra_repr(self):
        return super().extra_repr()

