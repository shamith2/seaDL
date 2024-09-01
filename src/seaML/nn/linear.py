from typing import Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

import string, random

import math
import mlx.core as mx
from mlx.nn import Module


@jaxtyped(typechecker=typechecker)
class Linear(Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device: Optional[mx.DeviceType] = None,
            dtype: Optional[mx.Dtype] = mx.float32
    ):
        '''
        A simple linear (affine) transformation
        '''
        super().__init__()

        self._use_einsum = True
        self.device = mx.gpu if not device else device

        self.in_features = in_features
        self.out_features = out_features

        scale = math.sqrt(1.0 / in_features)

        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_features, in_features),
            dtype=dtype
        )

        self.bias = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_features,),
            dtype=dtype
        ) if bias else None

    def __call__(
            self,
            x: mx.array
    ) -> mx.array:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        # y = xW.T + b
        if self._use_einsum:
            # subscript indices should not repeat with random.sample
            subscript = ' '.join(random.sample(string.ascii_lowercase, len(x.shape[:-1])))

            # "..." does work with mlx.core.einsum
            y = mx.einsum(
                subscript + " i, o i -> " + subscript + " o",
                x, self.weight,
                stream=self.device
            )

        else:
            y = mx.matmul(x, self.weight.T, stream=self.device)

        if self.bias is not None:
            y += self.bias

        return y

    def _extra_repr(self) -> str:
        return "in_features: {}, out_features: {}, bias_shape: {}".format(
            self.in_features, self.out_features, self.bias.shape if self.bias is not None else None
        )


@jaxtyped(typechecker=typechecker)
class Flatten(Module):
    def __init__(
            self,
            start_dim: int = 1,
            end_dim: int = -1,
            device: Optional[mx.DeviceType] = None
    ):
        super().__init__()

        self.device = mx.gpu if not device else device
        
        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(
            self,
            x: mx.array
    ) -> mx.array:
        '''
        Flatten out dimensions from start_dim to end_dim, inclusive of both
        '''
        in_shape = x.shape

        self.end_dim = len(in_shape) + self.end_dim if self.end_dim < 0 else self.end_dim

        shape = (in_shape[:self.start_dim] +
                (mx.prod(mx.array(in_shape[self.start_dim:self.end_dim+1]), stream=self.device).tolist(),) +
                in_shape[self.end_dim+1:])

        return x.reshape(*shape, stream=self.device)

    def _extra_repr(self) -> str:
        return super()._extra_repr()

