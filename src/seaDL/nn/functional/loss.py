from typing import Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from seaDL import Tensor
from seaDL.nn.functional import log_softmax


@jaxtyped(typechecker=typechecker)
def mse_loss(
        input: Tensor,
        target: Tensor,
        reduction: Optional[str] = 'mean'
):
    """
    Like torch.nn.functional.mse_loss
    """
    if reduction not in ['none', 'sum', 'mean']:
        raise ValueError("reduction has to be none, sum or mean")

    loss = (input - target) ** 2

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        return loss.sum()

    else:
        return loss.mean()


@jaxtyped(typechecker=typechecker)
def nll_loss(
        input: Tensor,
        target: Tensor,
        reduction: Optional[str] = 'mean'
):
    """
    Like torch.nn.functional.nll_loss
    """
    if reduction not in ['none', 'sum', 'mean']:
        raise ValueError("reduction has to be none, sum or mean")

    if 'int' not in target.dtype.value_as_str:
        raise ValueError("Expected target to be of type int or long")

    # for each row in input, pick the true classes as indicated by the target indices
    negative_loss_likelihood = -input[list(range(len(target))), target.tolist()]

    if reduction == 'none':
        return negative_loss_likelihood

    elif reduction == 'sum':
        return negative_loss_likelihood.sum()

    else:
        return negative_loss_likelihood.mean()


@jaxtyped(typechecker=typechecker)
def cross_entropy(
        input: Tensor,
        target: Tensor,
        reduction: Optional[str] = 'mean'
):
    """
    Like torch.nn.functional.cross_entropy
    """
    if reduction not in ['none', 'sum', 'mean']:
        raise ValueError("reduction has to be none, sum or mean")

    if 'int' not in target.dtype.value_as_str:
        raise ValueError("Expected target to be of type int or long")

    return nll_loss(log_softmax(input, dim=-1), target, reduction=reduction)

