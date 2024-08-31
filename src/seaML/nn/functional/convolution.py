from typing import Union, Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

import mlx.core as mx

from .utils import get_strides, _pair_value


@jaxtyped(typechecker=typechecker)
def pad1d_strided(
        x: mx.array,
        left: int,
        right: int,
        pad_value: float,
        device: mx.DeviceType
) -> mx.array:
    '''
    Return a new tensor with padding applied to the edges

    x: shape (batch, in_channels, width)

    Return: shape (batch, in_channels, left + right + width)
    '''
    b, c, w = x.shape

    out = mx.full(
        shape=(b, c, left + w + right),
        vals=pad_value,
        dtype=x.dtype,
        stream=device
    )

    out[..., left:left + w] = x

    return out


@jaxtyped(typechecker=typechecker)
def pad2d_strided(
        x: mx.array,
        top: int,
        bottom: int,
        left: int,
        right: int,
        pad_value: float,
        device: mx.DeviceType
) -> mx.array:
    '''
    Return a new tensor with padding applied to the edges

    x: shape (batch, in_channels, height, width)

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    b, c, h, w = x.shape

    out = mx.full(
        shape=(b, c, top + h + bottom, left + w + right),
        vals=pad_value,
        dtype=x.dtype,
        stream=device
    )

    out[..., top:top + h, left: left + w] = x

    return out


@jaxtyped(typechecker=typechecker)
def conv1d_strided(
    x: mx.array,
    weights: mx.array,
    stride: int = 1,
    padding: int = 0,
    device: Optional[mx.DeviceType] = None
) -> mx.array:
    '''
    Like torch.nn.functional.conv1d using bias=False

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    device = mx.gpu if not device else device

    pad_x = pad1d_strided(
        x,
        left=padding,
        right=padding,
        pad_value=0,
        device=device
    )

    x_dim0, x_dim1, x_dim2 = get_strides(pad_x.shape, device)

    b, ic, w = pad_x.shape
    _, w_ic, kw = weights.shape

    if ic != w_ic:
        raise ValueError("The number of input features in x should be equal to the number of input features in weights")

    x_prime = mx.as_strided(
        pad_x,
        shape=(b, ic, 1 + (w - kw) // stride, kw),
        strides=(x_dim0.item(), x_dim1.item(), x_dim2.item() * stride, x_dim2.item()),
        stream=device
    )

    x_conv = mx.einsum(
        "b i w k, o i k -> b o w",
        x_prime, weights,
        stream=device
    )

    return x_conv


@jaxtyped(typechecker=typechecker)
def conv2d_strided(
    x: mx.array,
    weights: mx.array,
    stride: Union[int, tuple[int, int]] = 1,
    padding: Union[int, tuple[int, int]] = 0,
    device: Optional[mx.DeviceType] = None
) -> mx.array:
    '''
    Like torch.nn.functional.conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    device = mx.gpu if not device else device

    stride, padding = _pair_value(stride), _pair_value(padding)

    pad_x = pad2d_strided(
        x,
        top=padding[0],
        bottom=padding[0],
        left=padding[1],
        right=padding[1],
        pad_value=0,
        device=device
    )

    x_dimb, x_dimc, x_dimh, x_dimw = get_strides(pad_x.shape, device)

    b, ic, h, w = pad_x.shape
    _, w_ic, kh, kw = weights.shape

    if ic != w_ic:
        raise ValueError("The number of input features in x should be equal to the number of input features in weights")

    x_prime = mx.as_strided(
        pad_x,
        shape=(b, ic, (h - kh) // stride[0] + 1, (w - kw) // stride[1] + 1, kh, kw),
        strides=(x_dimb.item(), x_dimc.item(), x_dimh.item() * stride[0], x_dimw.item() * stride[1], x_dimh.item(), x_dimw.item()),
        stream=device
    )

    x_conv = mx.einsum(
        "b i h w y x, o i y x -> b o h w",
        x_prime, weights,
        stream=device
    )

    return x_conv

