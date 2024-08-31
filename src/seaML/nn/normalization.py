from typing import Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from einops.array_api import rearrange
import mlx.core as mx
from mlx.nn import Module


@jaxtyped(typechecker=typechecker)
class BatchNorm2d(Module):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-05,
            momentum: float = 0.1,
            device: Optional[mx.DeviceType] = None
    ):
        '''
        Like nn.BatchNorm2d with track_running_stats=True and affine=True
        '''
        super().__init__()

        self.device = mx.gpu if not device else device

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # gamma and beta
        self.weight = mx.ones(shape=(num_features,))
        self.bias = mx.zeros(shape=(num_features,))

        # buffers
        self.running_mean = mx.zeros(shape=(num_features,))
        self.running_var = mx.ones(shape=(num_features,))
        self.num_batches_tracked = mx.zeros(shape=(1,))

        # freeze buffers since they are not required for gradient computation
        self.freeze(keys=['running_mean', 'running_var', 'num_batches_tracked'])

    def __call__(
            self,
            x: mx.array
    ) -> mx.array:
        '''
        Normalize each channel/feature

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        if self.training:
            mean = mx.mean(x, axis=(0, 2, 3), keepdims=True, stream=self.device)
            var = mx.var(x, ddof=0, axis=(0, 2, 3), keepdims=True, stream=self.device)
            moving_var = mx.var(x, ddof=1, axis=(0, 2, 3), keepdims=True, stream=self.device)

            self.running_mean = ((1 - self.momentum) * self.running_mean +
                                 mean.reshape((self.num_features,)) * self.momentum)

            self.running_var = ((1 - self.momentum) * self.running_mean +
                                moving_var.reshape((self.num_features,)) * self.momentum)

            self.num_batches_tracked += 1

        else:
            mean = rearrange(self.running_mean, "channels -> 1 channels 1 1")
            var = rearrange(self.running_var, "channels -> 1 channels 1 1")

        weight = rearrange(self.weight, "channels -> 1 channels 1 1")
        bias = rearrange(self.bias, "channels -> 1 channels 1 1")

        y = (x - mean) / mx.sqrt(var + self.eps, stream=self.device)

        out = y * weight + bias

        return out

    def _extra_repr(self) -> str:
        return super()._extra_repr()

