from typing import Iterable
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

import functools
from contextlib import contextmanager

from .config import config


class GradientConfig:
    # global flag to setup computational graph
    # and compute gradients 
    enable_grad: bool = True


@contextmanager
def no_grad():
    try:
        GradientConfig.enable_grad = False
        yield

    finally:
        GradientConfig.enable_grad = True


@jaxtyped(typechecker=typechecker)
class Device:
    def __init__(
            self,
            device: str
    ):
        self.value = config.get_device(device)


@jaxtyped(typechecker=typechecker)
class DataType:
    def __init__(
            self,
            dtype: str = 'float32'
    ):
        self.value_as_str = dtype

        # to accomodate for dtype strings
        # like "mlx.core.float32"
        if 'float32' in dtype.lower():
            self.value = config.backend.float32

        elif 'float16' in dtype.lower():
            self.value = config.backend.float16

        elif 'int32' in dtype.lower():
            self.value = config.backend.int32

        elif 'int64' in dtype.lower():
            self.value = config.backend.int64

        else:
            raise ValueError("dtype is not a valid datatype or is not supported")


@jaxtyped(typechecker=typechecker)
def prod(
    array: Iterable
): 
    return functools.reduce((lambda x, y: x * y), array)

