from typing import Union
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

import mlx.core as mx


@jaxtyped(typechecker=typechecker)
def _get_strides(
        shape: tuple,
        device: mx.DeviceType
):
    """
    Like torch.Tensor.stride

    If shape of tensor is (2, 16, 32),
    then, the stride in dim 0 = 1 (since the elements in dim 0 are consecutive in memory),
    dim 1 = 32 (since elements in dim 1 are 32 elements apart) and
    dim 2 = 32 * 16 (since elements in dim 2 are 16 blocks apart where each block is 32 elements),
    so function will return (512, 32, 1)
    """
    strides = mx.array([1] * len(shape))

    strides[:-1] = mx.cumprod(mx.array(shape[::-1]), stream=device)[::-1][1:]

    # strides = mx.concatenate((strides, mx.array([1])), axis=0, stream=device)

    return strides


@jaxtyped(typechecker=typechecker)
def _pair_value(
        v: Union[int, tuple[int, int]]
) -> tuple[int, int]:
    '''
    Convert v to a pair of int, if it isn't already
    '''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return v

    elif isinstance(v, int):
        return (v, v)

    raise ValueError(v)

