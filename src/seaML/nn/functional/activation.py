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
    # return mx.maximum(x, 0.0, stream=device)

    return x.maximum(0.0)

