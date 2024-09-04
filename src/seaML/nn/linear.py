from typing import Optional
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

import string, random

import math
import mlx.core as mx

from .base import Tensor, Parameter, Module
from ..utils import prod


@jaxtyped(typechecker=typechecker)
class Linear(Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            einsum_skipdims: int = 0,
            device: Optional[mx.DeviceType] = None,
            dtype: Optional[mx.Dtype] = mx.float32
    ):
        '''
        A simple linear (affine) transformation
        '''
        super().__init__()

        # need to get input for einsum notation subscript
        # since "..." does work with mlx.core.einsum,
        # atleast not like numpy.einsum
        if einsum_skipdims:
            self.use_einsum = True
            self.einsum_skipdims = einsum_skipdims

        else:
            self.use_einsum = False

        self.device = mx.gpu if not device else device

        self.in_features = in_features
        self.out_features = out_features

        scale = math.sqrt(1.0 / in_features)

        self.weight = Parameter(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(out_features, in_features),
                dtype=dtype
            )
        )

        if bias:
            self.bias = Parameter(
                mx.random.uniform(
                    low=-scale,
                    high=scale,
                    shape=(out_features,),
                    dtype=dtype
                )
            )

        else:
            self.bias = None


    def __call__(
            self,
            x: Tensor
    ) -> mx.array:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        # y = xW.T + b
        if self.use_einsum:
            # subscript indices should not repeat with random.sample
            subscript = ' '.join(random.sample(string.ascii_lowercase, self.einsum_skipdims))

            y = x.einsum(
                subscript + " i, o i -> " + subscript + " o",
                self.weight
            )

        else:
            y = x.matmul(self.weight.transpose())

        if self.bias is not None:
            y = y + self.bias

        return y


    def extra_repr(self) -> str:
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
            x: Tensor
    ) -> Tensor:
        '''
        Flatten out dimensions from start_dim to end_dim, inclusive of both
        '''
        in_shape = x.shape

        self.end_dim = len(in_shape) + self.end_dim if self.end_dim < 0 else self.end_dim

        shape = (in_shape[:self.start_dim] +
                (prod(in_shape[self.start_dim:self.end_dim+1]),) +
                in_shape[self.end_dim+1:])

        # return x.reshape(*shape, stream=self.device)
        return x.reshape(shape)


    def extra_repr(self) -> str:
        return super().extra_repr()

