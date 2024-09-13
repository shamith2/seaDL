from typing import Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from ..engine import Tensor, Device
from .base import Module
from .functional import relu, softmax, log_softmax


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


@jaxtyped(typechecker=typechecker)
class Softmax(Module):
    def __init__(
            self,
            dim: int = -1,
            device: Optional[Device] = None
    ):
        super().__init__()

        self.dim = dim


    def __call__(
            self,
            x: Tensor
    ) -> Tensor:
        """
        Like torch.nn.Softmax using functional softmax
        """
        return softmax(x, dim=self.dim)


    def extra_repr(self):
        return super().extra_repr()


@jaxtyped(typechecker=typechecker)
class LogSoftmax(Module):
    def __init__(
            self,
            dim: int = -1,
            device: Optional[Device] = None
    ):
        super().__init__()

        self.dim = dim


    def __call__(
            self,
            x: Tensor
    ) -> Tensor:
        """
        Like torch.nn.LogSoftmax using functional log_softmax
        """
        return log_softmax(x, dim=self.dim)


    def extra_repr(self):
        return super().extra_repr()

