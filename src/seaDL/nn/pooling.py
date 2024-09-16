from typing import Union, Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from seaDL import Tensor, Device
from seaDL.nn.functional import max_pool2d, avg_pool2d
from seaDL.utils import _pair_value

from .base import Module


@jaxtyped(typechecker=typechecker)
class MaxPool2d(Module):
    def __init__(
            self,
            kernel_size: Union[int, tuple[int, int]],
            stride: Optional[Union[int, tuple[int, int]]] = None,
            padding: Union[int, tuple[int, int]] = 1,
            device: Optional[Device] = None
    ):
        """
        Like torch.nn.MaxPool2d with dilation = 1 and ceil_mode = False
        """
        super().__init__()
        
        self.kernel_size = _pair_value(kernel_size)
        self.stride = stride
        self.padding = _pair_value(padding)


    def __call__(
            self,
            x: Tensor
    ) -> Tensor:
        '''call the functional version of max_pool2d'''
        out = max_pool2d(
            x,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

        return out


    def _extra_repr(self) -> str:
        return super().extra_repr()


class AveragePool2d(Module):
    def __init__(
            self,
            device: Optional[Device] = None
    ):
        super().__init__()


    def __call__(
            self,
            x: Tensor
    ) -> Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        return avg_pool2d(x)


    def extra_repr(self) -> str:
        return super().extra_repr()

