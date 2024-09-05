from typing import Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from ...base import Tensor, Device


@jaxtyped(typechecker=typechecker)
def relu(
        x: Tensor,
        device: Optional[Device] = None
) -> Tensor:
    """
    Like torch.nn.functional.relu
    """
    return x.maximum(0.0)

