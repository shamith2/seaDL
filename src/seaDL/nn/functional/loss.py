from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from seaDL import Tensor


@jaxtyped(typechecker=typechecker)
def mse_loss(
        input: Tensor,
        target: Tensor
):
    """
    Similar to torch.nn.functional.mse_loss with reduction = 'mean'
    """
    return ((input - target) ** 2).mean()

