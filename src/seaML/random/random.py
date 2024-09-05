# Wrapper fucntions for relevant random library functions from backend

from typing import Any, Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from ..config import config, ArrayType
from ..base import Tensor, DataType


# @jaxtyped(typechecker=typechecker)
def uniform(
        shape: tuple,
        low: float = 0.0,
        high: float = 1.0,
        dtype: Optional[DataType] = DataType('float32')
):
    data = config.backend.random.uniform(
                low=low,
                high=high,
                shape=shape
            )

    return Tensor(
        data=data,
        dtype=dtype
    )


@jaxtyped(typechecker=typechecker)
def normal(
        shape: tuple,
        mean: float = 0.0,
        scale: float = 1.0,
        dtype: Optional[DataType] = DataType('float32')
):
    data=config.backend.random.normal(
            shape=shape,
            loc=mean,
            scale=scale
        )

    return Tensor(
        data=data,
        dtype=dtype
    )

