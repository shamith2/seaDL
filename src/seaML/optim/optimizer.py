from typing import Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

import mlx.core as mx

from ..base import Device
from ..nn import Module


@jaxtyped(typechecker=typechecker)
class SGD:
    def __init__(
        self,
        model: Module,
        lr: float,
        momentum: float = 0.0,
        dampening: Optional[float] = 0.0,
        weight_decay: Optional[float] = 0.0,
        nesterov: Optional[bool] = False,
        device: Optional[Device] = None
    ):
        '''
        Implements SGD with momentum

        Like the PyTorch version, but assume nesterov=False, maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD

        '''
        if nesterov and (momentum <= 0 or dampening):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        train_params: dict = model.trainable_parameters()

        parameters = []

        for key, value in train_params.items():
            if isinstance(value, (list, tuple)):
                print(value[0], len(value[0]))
        vv
        for name, module in model.named_modules():
            print(name, module)
            print(train_params.get(name))
            cc
        
        cc

        # if value is a list, key is a list of layers
        if isinstance(value, (list, tuple)):
            for v in value:
                print(type(v), list(v.keys()))
                cc

        cc

        # turn params into a list because it might be a generator
        self.params = list(params.view(stream=self.device))

        self.lr = lr
        self.mu = momentum
        self.lmda = weight_decay
        self.tau = dampening

        self.nesterov = nesterov

        # set number of steps to zero initially
        self.t = 0

        # set gradients for paramters to zero initially
        self.gs = [mx.zeros_like(p, stream=self.device) for p in self.params]

    def zero_grad(self) -> None:
        '''
        Zeros all gradients of the parameters in `self.params`
        '''
        for param in self.params:
            param.grad = mx.zeros_like(param, stream=self.device)

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
                g_t += self.lmda * p

            # if momentum != 0
            if self.mu:
                if self.t > 0:
                    b_t = self.mu * g + (1 - self.tau) * g_t

                else:
                    b_t = g_t

                if self.nesterov:
                    g_t += self.mu * b_t

                else:
                    g_t = b_t

            # update parameters
            self.params[i] -= self.lr * g_t

            # store updated gradients
            self.gs[i] = g_t

        # update step count
        self.t += 1

    def __repr__(self) -> str:
        return "SGD(lr={}, momentum={}, weight_decay={})".format(
            self.lr, self.mu, self.lmda
        )

