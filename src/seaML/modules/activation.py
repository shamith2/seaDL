from typing import Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

import mlx.core as mx


@jaxtyped(typechecker=typechecker)
def relu(
        x: mx.array,
        inplace: bool,
        device: Optional[mx.DeviceType] = None
) -> mx.array:
    """
    Like torch.nn.functional.relu
    """
    if inplace:
        # mlx does not yet support boolean indices
        indices = [i for i, e in enumerate(x < 0.0) if e]

        x[indices] = 0.0
        return x

    else:
        device = mx.gpu if not device else device

        return mx.maximum(x, 0.0, stream=device)

