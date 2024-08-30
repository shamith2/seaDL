from typing import Union, Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

import mlx.core as mx
import mlx.nn as nn

from ..modules.utils import _pair_value
from ..modules.pooling import maxpool2d_strided, averagepool2d


@jaxtyped(typechecker=typechecker)
class MaxPool2d(nn.Module):
    def __init__(
            self,
            kernel_size: Union[int, tuple[int, int]],
            stride: Optional[Union[int, tuple[int, int]]] = None,
            padding: Union[int, tuple[int, int]] = 1,
            device: Optional[mx.DeviceType] = None
    ):
        """
        Like torch.nn.MaxPool2d with dilation = 1 and ceil_mode = False
        """
        super().__init__()

        self.device = mx.gpu if not device else device
        
        self.kernel_size = _pair_value(kernel_size)
        self.stride = stride
        self.padding = _pair_value(padding)

    def __call__(
            self,
            x: mx.array
    ) -> mx.array:
        '''Call the functional version of max_pool2d'''
        out = maxpool2d_strided(
            x,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            device=self.device
        )

        return out

    def _extra_repr(self) -> str:
        return super()._extra_repr()


class AveragePool2d(nn.Module):
    def __init__(
            self,
            device: Optional[mx.DeviceType] = None
    ):
        super().__init__()

        self.device = mx.gpu if not device else device

    def __call__(
            self,
            x: mx.array
    ) -> mx.array:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        return averagepool2d(
            x,
            device=self.device
        )

    def _extra_repr(self) -> str:
        return super()._extra_repr()

