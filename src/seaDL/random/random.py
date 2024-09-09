# Wrapper fucntions for relevant random library functions from backend

from typing import Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from ..config import config
from ..base import Tensor, DataType


# @jaxtyped(typechecker=typechecker)
def uniform(
        size: tuple,
        low: float = 0.0,
        high: float = 1.0,
        dtype: Optional[DataType] = DataType('float32')
):
    if config.is_backend_mlx():
        data = config.backend.random.uniform(
                    low=low,
                    high=high,
                    shape=size
                )

    elif config.is_backend_numpy():
        rng = config.backend.random.default_rng()

        data = rng.uniform(
                    low=low,
                    high=high,
                    size=size
                )

    return Tensor(
        data=data,
        dtype=dtype
    )


@jaxtyped(typechecker=typechecker)
def normal(
        size: tuple,
        mean: float = 0.0,
        scale: float = 1.0,
        dtype: Optional[DataType] = DataType('float32')
):
    if config.is_backend_mlx():
        data=config.backend.random.normal(
                shape=size,
                loc=mean,
                scale=scale
            )

    elif config.is_backend_numpy():
        rng = config.backend.random.default_rng()

        data=rng.normal(
                loc=mean,
                scale=scale,
                size=size
            )

    return Tensor(
        data=data,
        dtype=dtype
    )

