from typing import Union, Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

import math
import mlx.core as mx

from .base import Module, Parameter
from .functional import conv1d, conv2d
from ..utils import _pair_value


@jaxtyped(typechecker=typechecker)
class Conv1d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        device: Optional[mx.DeviceType] = None,
        dtype: mx.Dtype = mx.float32
    ):
        '''
        Like torch.nn.Conv1d with dilation = 1 and groups = 1
        '''
        super().__init__()

        self.device = mx.gpu if not device else device

        self.stride = stride
        self.padding = padding
        
        k = 1.0 / (in_channels * kernel_size)
        scale = -math.sqrt(k)
        
        self.weight = Parameter(
                mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(out_channels, in_channels, kernel_size),
                dtype=dtype
            )
        )

        if bias:
            self.bias = Parameter(
                    mx.random.uniform(
                    low=-scale,
                    high=scale,
                    shape=(out_channels,),
                    dtype=dtype
                )
            )

        else:
            self.bias = None

    def __call__(
            self,
            x: mx.array
    ) -> mx.array:
        '''Apply the functional conv1d'''
        out = conv1d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            device=self.device
        )

        if self.bias is not None:
            out += self.bias.expand_dims(dim=(0, 2), device=self.device)

        return out

    def _extra_repr(
            self
    ) -> str:
        return ("Weight shape: {}, Kernel Size: {}, Stride: {}, Padding: {}, Parameters: {}"
                .format(self.weight.shape, self.weight.shape[-1],
                        self.stride, self.padding, self.weight.size))


@jaxtyped(typechecker=typechecker)
class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]] = 1,
        padding: Union[int, tuple[int, int]] = 0,
        bias: bool = True,
        device: Optional[mx.DeviceType] = None,
        dtype: mx.Dtype = mx.float32
    ):
        '''
        Like torch.nn.Conv2d with dilation = 1 and groups = 1
        '''
        super().__init__()

        self.device = mx.gpu if not device else device

        self.kh, self.kw = _pair_value(kernel_size)
        self.stride = _pair_value(stride)
        self.padding = _pair_value(padding)
        
        k = 1.0 / (in_channels * self.kh * self.kw)
        scale = -math.sqrt(k)
        
        self.weight = Parameter(
                mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(out_channels, in_channels, self.kh, self.kw),
                dtype=dtype
            )
        )

        if bias:
            self.bias = Parameter(
                    mx.random.uniform(
                    low=-scale,
                    high=scale,
                    shape=(out_channels,),
                    dtype=dtype
                )
            )

        else:
            self.bias = None

    def __call__(
            self,
            x: mx.array
    ) -> mx.array:
        '''Apply the functional conv2d'''
        out = conv2d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            device=self.device
        )

        if self.bias is not None:
            out += self.bias.expand_dims(dim=(0, 2, 3), device=self.device)

        return out

    def _extra_repr(
            self
    ) -> str:
        return ("Weight shape: {}, Kernel Size: {}, Stride: {}, Padding: {}, Parameters: {}"
                .format(self.weight.shape, self.weight.shape[1:2],
                        self.stride, self.padding, self.weight.size))

