from typing import Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from seaDL import Tensor, Device


@jaxtyped(typechecker=typechecker)
def relu(
        x: Tensor,
        device: Optional[Device] = None
) -> Tensor:
    """
    Like torch.nn.functional.relu
    """
    return x.maximum(0.0)


@jaxtyped(typechecker=typechecker)
def softmax(
        x: Tensor,
        dim: int,
        device: Optional[Device] = None
) -> Tensor:
    """
    Like torch.nn.functional.softmax
    """
    # x - max(x) is used for numerical stability
    x_prime = x - x.max(dim=dim, keepdim=True)

    exp_values = x_prime.exp()

    return exp_values / exp_values.sum(dim=dim, keepdim=True)


@jaxtyped(typechecker=typechecker)
def log_softmax(
        x: Tensor,
        dim: int,
        device: Optional[Device] = None
) -> Tensor:
    """
    Like torch.nn.functional.log_softmax
    """
    return softmax(x, dim=dim).log()

