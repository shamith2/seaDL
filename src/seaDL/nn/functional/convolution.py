from typing import Union, Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from ...engine import Tensor, Device, full
from ...nn.base import Parameter
from ...utils import _pair_value


@jaxtyped(typechecker=typechecker)
def pad1d(
        x: Tensor,
        left: int,
        right: int,
        pad_value: float,
        device: Optional[Device] = None
) -> Tensor:
    '''
    Return a new tensor with padding applied to the edges

    x: shape (batch, in_channels, width)

    Return: shape (batch, in_channels, left + right + width)
    '''
    b, c, w = x.shape

    out = full(
        shape=(b, c, left + w + right),
        fill_value=pad_value,
        dtype=x.dtype
    )

    # slicing similar to out[..., left:(left + w)]
    return out.set_slice((Ellipsis, slice(left, left + w)), x)


@jaxtyped(typechecker=typechecker)
def pad2d(
        x: Tensor,
        top: int,
        bottom: int,
        left: int,
        right: int,
        pad_value: float,
        device: Optional[Device] = None
) -> Tensor:
    '''
    Return a new tensor with padding applied to the edges

    x: shape (batch, in_channels, height, width)

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    b, c, h, w = x.shape

    out = full(
        shape=(b, c, top + h + bottom, left + w + right),
        fill_value=pad_value,
        dtype=x.dtype
    )

    # slicing similar to out[..., top:top + h, left: left + w]
    return out.set_slice((Ellipsis, slice(top, top + h),
                          slice(left, left + w)), x)


@jaxtyped(typechecker=typechecker)
def conv1d(
    x: Tensor,
    weights: Union[Tensor, Parameter],
    stride: int = 1,
    padding: int = 0,
    device: Optional[Device] = None
) -> Tensor:
    '''
    Like torch.nn.functional.conv1d using bias=False

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    pad_x = pad1d(
        x,
        left=padding,
        right=padding,
        pad_value=0
    )

    b, ic, w = pad_x.shape
    _, w_ic, kw = weights.shape

    if ic != w_ic:
        raise ValueError("The number of input features in x should be equal to the number of input features in weights")

    x_dim0, x_dim1, x_dim2 = pad_x.stride

    x_prime = pad_x.as_strided(
        shape=(b, ic, 1 + (w - kw) // stride, kw),
        strides=(x_dim0, x_dim1, x_dim2 * stride, x_dim2)
    )

    return x_prime.einsum(
        "b i w k, o i k -> b o w",
        weights
    )


@jaxtyped(typechecker=typechecker)
def conv2d(
    x: Tensor,
    weights: Union[Tensor, Parameter],
    stride: Union[int, tuple[int, int]] = 1,
    padding: Union[int, tuple[int, int]] = 0,
    device: Optional[Device] = None
) -> Tensor:
    '''
    Like torch.nn.functional.conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    stride, padding = _pair_value(stride), _pair_value(padding)

    pad_x = pad2d(
        x,
        top=padding[0],
        bottom=padding[0],
        left=padding[1],
        right=padding[1],
        pad_value=0
    )

    b, ic, h, w = pad_x.shape
    _, w_ic, kh, kw = weights.shape

    if ic != w_ic:
        raise ValueError("The number of input features in x should be equal to the number of input features in weights")

    x_dimb, x_dimc, x_dimh, x_dimw = pad_x.stride

    x_prime = pad_x.as_strided(
        shape=(b, ic, (h - kh) // stride[0] + 1, (w - kw) // stride[1] + 1, kh, kw),
        strides=(x_dimb, x_dimc, x_dimh * stride[0],
                 x_dimw * stride[1], x_dimh, x_dimw)
    )

    return x_prime.einsum(
        "b i h w y x, o i y x -> b o h w",
        weights
    )

