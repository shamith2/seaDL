from typing import Optional, Iterable
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from ..engine import zeros_like, no_grad
from ..nn import Parameter


@jaxtyped(typechecker=typechecker)
class SGD:
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float,
        momentum: float = 0.0,
        dampening: Optional[float] = 0.0,
        weight_decay: Optional[float] = 0.0,
        nesterov: Optional[bool] = False
    ):
        '''
        Implements SGD with momentum

        Like the PyTorch version, but assume nesterov=False, maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD

        '''
        if nesterov and (momentum <= 0 or dampening):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        # turn params into a list (because it might be a generator)
        self.params = list(params)

        self.lr = lr
        self.mu = momentum
        self.lmda = weight_decay
        self.tau = dampening

        self.nesterov = nesterov

        # set number of steps to zero initially
        self.t = 0

        # model parameter gradients
        self.gs = [zeros_like(param) for param in self.params]


    def zero_grad(self):
        for param in self.params:
            param.zero_grad()


    @no_grad()
    def step(self) -> None:
        '''
        Performs a single optimization step of the SGD algorithm

        Comparision to PyTorch documentation:
            1. g_t -> g_t : updated gradient of parameter i for current step
            2. theta_(t-1) -> p or self.params[i] : parameter i
            3. b_(t-1) -> g: exisiting gradient of paramter i from previous step

        '''
        for i, (p, g) in enumerate(zip(self.params, self.gs)):
            # g_t = grad_fn(theta_(t-1))
            g_t = p.grad

            # if weight decay != 0
            if self.lmda:
                g_t += (p * self.lmda)

            # if momentum != 0
            if self.mu:
                if self.t > 0:
                    b_t = g * self.mu + g_t * (1 - self.tau)

                else:
                    b_t = g_t

                if self.nesterov:
                    g_t += (b_t * self.mu)

                else:
                    g_t = b_t

            # update parameters
            # needs to be in-place since the model
            # parameters need to be directly updated
            self.params[i] -= g_t * self.lr

            # store updated gradients
            self.gs[i] = g_t

        # update step count
        self.t += 1


    def __repr__(self) -> str:
        return "SGD(lr={}, momentum={}, weight_decay={})".format(
            self.lr, self.mu, self.lmda
        )

