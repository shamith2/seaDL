from typing import Union, Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

import mlx.core as mx

from .utils import _get_strides, _pair_value
from .convolution import pad2d_strided


@jaxtyped(typechecker=typechecker)
def maxpool2d_strided(
    x: mx.array,
    kernel_size: Union[int, tuple[int, int]],
    stride: Optional[Union[int, tuple[int, int]]] = None,
    padding: Union[int, tuple[int, int]] = 0,
    device: Union[mx.DeviceType, None] = None
) -> mx.array:
    '''
    Like torch.nn.functional.max_pool2d

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, output_height, output_width)
    '''
    device = mx.gpu if not device else device

    kh, kw = _pair_value(kernel_size)
    stride = _pair_value(stride) if stride else _pair_value(kernel_size)
    padding = _pair_value(padding)

    pad_x = pad2d_strided(
        x,
        top=padding[0],
        bottom=padding[0],
        left=padding[1],
        right=padding[1],
        pad_value=-mx.inf,
        device=device
    )

    b, ic, h, w = pad_x.shape
    x_dimb, x_dimc, x_dimh, x_dimw = _get_strides(pad_x.shape, device)

    x_prime = mx.as_strided(
        pad_x,
        shape=(b, ic, (h - kh) // stride[0] + 1, (w - kw) // stride[1] + 1, kh, kw),
        strides=(x_dimb.item(), x_dimc.item(), x_dimh.item() * stride[0], x_dimw.item() * stride[1], x_dimh.item(), x_dimw.item()),
        stream=device
    )

    x_max = mx.max(x_prime, axis=(-2, -1))

    return x_max


@jaxtyped(typechecker=typechecker)
def averagepool2d(
    x: mx.array,
    device: Union[mx.DeviceType, None] = None
) -> mx.array:
    """
    x: shape (batch, channels, height, width)
    Return: shape (batch, channels)
    """
    out = mx.mean(x, axis=(-2, -1), stream=device)

    return out

