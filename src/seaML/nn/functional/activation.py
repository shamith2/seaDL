from typing import Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

import mlx.core as mx

from ...nn.base import Tensor


@jaxtyped(typechecker=typechecker)
def relu(
        x: Tensor,
        device: Optional[mx.DeviceType] = None
) -> Tensor:
    """
    Like torch.nn.functional.relu
    """
    return x.maximum(0.0)

