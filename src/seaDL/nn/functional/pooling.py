from typing import Union, Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from ...config import config
from ...base import Tensor, Device
from .convolution import pad2d
from ...utils import _pair_value


@jaxtyped(typechecker=typechecker)
def maxpool2d(
    x: Tensor,
    kernel_size: Union[int, tuple[int, int]],
    stride: Optional[Union[int, tuple[int, int]]] = None,
    padding: Union[int, tuple[int, int]] = 0,
    device: Optional[Device] = None
) -> Tensor:
    '''
    Like torch.nn.functional.max_pool2d

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, output_height, output_width)
    '''
    kh, kw = _pair_value(kernel_size)
    stride = _pair_value(stride) if stride else _pair_value(kernel_size)
    padding = _pair_value(padding)

    pad_x = pad2d(
        x,
        top=padding[0],
        bottom=padding[0],
        left=padding[1],
        right=padding[1],
        pad_value=-config.backend.inf
    )

    b, ic, h, w = pad_x.shape

    x_dimb, x_dimc, x_dimh, x_dimw = pad_x.stride

    x_prime = pad_x.as_strided(
        shape=(b, ic, (h - kh) // stride[0] + 1, (w - kw) // stride[1] + 1, kh, kw),
        strides=(x_dimb, x_dimc, x_dimh * stride[0], x_dimw * stride[1], x_dimh, x_dimw)
    )

    return x_prime.max(dim=(-2, -1))


@jaxtyped(typechecker=typechecker)
def averagepool2d(
    x: Tensor,
    device: Optional[Device] = None
) -> Tensor:
    """
    x: shape (batch, channels, height, width)
    Return: shape (batch, channels)
    """
    return x.mean(dim=(-2, -1))

