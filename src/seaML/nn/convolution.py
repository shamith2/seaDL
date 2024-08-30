from typing import Union
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

import math
import mlx.core as mx
import mlx.nn as nn

from ..modules.utils import _pair_value
from ..modules.convolution import conv2d_strided


@jaxtyped(typechecker=typechecker)
class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]] = 1,
        bias: bool = True,
        padding: Union[int, tuple[int, int]] = 0,
        device: Union[mx.DeviceType, None] = None,
        dtype: mx.Dtype = mx.float32
    ):
        '''
        Same as torch.nn.Conv2d with dilation = 1 and groups = 1
        '''
        super().__init__()

        self.device = mx.gpu if not device else device

        self.kh, self.kw = _pair_value(kernel_size)
        self.stride = _pair_value(stride)
        self.padding = _pair_value(padding)
        
        k = 1.0 / (in_channels * self.kh * self.kw)
        scale = -math.sqrt(k)
        
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, in_channels, self.kh, self.kw),
            dtype=dtype
        )

        if bias:
            self.bias = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(out_channels,),
                dtype=dtype
            )

        else:
            self.bias = None

    def __call__(
            self,
            x: mx.array
    ) -> mx.array:
        '''Apply the functional conv2d'''
        out = conv2d_strided(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            device=self.device
        )

        if self.bias is not None:
            out = out + mx.expand_dims(self.bias, axis=(0, 2, 3), stream=self.device)

        return out

    def _extra_repr(
            self
    ) -> str:
        return ("Weight shape: {}, Kernel Size: {}, Stride: {}, Padding: {}, Parameters: {}"
                .format(self.weight.shape, self.weight.shape[1:2],
                        self.stride, self.padding, self.weight.size))

