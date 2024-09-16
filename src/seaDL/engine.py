from typing import Any, Union, Optional, Self, Callable, Iterable
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

import copy
import collections
import math

from .config import config, ArrayType
from .base import DataType, GradientConfig, prod


@jaxtyped(typechecker=typechecker)
class Tensor:
    """
    Class that represents a Tensor datatype

    Each node is a Tensor in the computational graph

    Incorporates lazy computation and automatic differentiation
    """
    def __init__(
            self,
            data: Iterable = (),
            dtype: Optional[DataType] = None,
            requires_grad: Optional[bool] = False
    ):
        data = () if data is None else data

        # value of node: can be result of an operation or constant value
        if not isinstance(data, ArrayType):
            self.data = config.Array(data)

        else:
            self.data = data

        # dtype of Tensor
        if dtype is not None:
            self.astype(dtype)

        else:
            if config.is_backend_numpy() and 'float64' in str(self.data.dtype):
                self.data_type = DataType('float32')
                self.astype(self.data_type)

            else:
                self.data_type = DataType(str(self.data.dtype))

        # operation in the computational graph
        # that connects this Tensor
        self.operation: Optional[Operation] = None

        # index of Tensor in the Tensor's operation output
        # if output is a list of Tensors
        self.output_index: Optional[int] = None

        # whether to compute gradient for the node
        self.requires_grad = requires_grad and GradientConfig.enable_grad

        # gradient with respect to this node
        self.zero_grad()


    def __array__(self):
        """
        For converting to numpy array,
        using numpy.array()
        """
        return NotImplementedError


    def __deepcopy__(self, memo):
        """
        Support copy.deepcopy
        """
        copied_tensor = Tensor(
            data=copy.deepcopy(self.data, memo),
            dtype=self.data_type,
            requires_grad=self.requires_grad
        )

        copied_tensor.grad = copy.deepcopy(self.grad, memo) if self.grad is not None else None

        return copied_tensor


    def numel(self) -> int:
        """
        Number of elements in Tensor
        """
        return self.data.size


    @property
    def ndim(self) -> int:
        """
        Number of dimensions of Tensor
        """
        return self.data.ndim


    def __getitem__(
            self,
            indices: Iterable
    ):
        """
        Support Tensor indexing

        Retrieve slices of elements from Tensor
        """
        def _grad_fn(gradient: ArrayType, *inputs: tuple[ArrayType]):
            input_gradient = config.backend_op('zeros_like', inputs[0])
            input_gradient[indices] = gradient

            return (input_gradient,)

        _operation = lambda x: x[indices]


        if GradientConfig.enable_grad and self.requires_grad:
            node = Operation(
                name='get_slice',
                operation=_operation,
                inputs=(self,),
                grad_fn=_grad_fn
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=True
            )

            # result of node
            result.operation = node

        else:
            result = Tensor(
                data=_operation(self.data),
                dtype=self.data_type,
                requires_grad=False
            )

        return result


    def __setitem__(
            self,
            indices: Iterable,
            value: Any
    ):
        """
        Support Tensor indexing

        Set slices of elements from Tensor to value

        This operation fires the computational graph
        """
        if self.numel() == 0:
            self.data = self.operation.fire()

        self.data[indices] = value.data if isinstance(value, Tensor) else value


    @property
    def shape(self) -> tuple:
        if self.numel() == 0:
            # when .shape is called,
            # if Tensor has not been computed,
            # compute the Tensor
            if self.operation is not None:
                self.data = self.operation.fire()

        return self.data.shape


    @property
    def dtype(self):
        return self.data_type


    @property
    def itemsize(self):
        return self.data.itemsize


    def __len__(self):
        return len(self.data)


    def tolist(self):
        return self.data.tolist()


    def totuple(self):
        return tuple(self.data.tolist())


    @property
    def stride(self):
        """
        Like torch.Tensor.stride

        If shape of tensor is (2, 16, 32),
        then, the stride in dim 0 = 1
        (since the elements in dim 0 are consecutive in memory),
        dim 1 = 32 (since elements in dim 1 are 32 elements apart) and
        dim 2 = 32 * 16 (since elements in dim 2 are 16 blocks apart
        where each block is 32 elements),
        so function will return (512, 32, 1)
        """
        _shape = self.data.shape

        strides = config.Array([1] * len(_shape))
        strides[:-1] = config.backend.cumprod(
            config.Array(_shape[::-1])
        )[::-1][1:]

        if config.is_backend_numpy():
            strides = strides * self.itemsize

        return tuple(strides.tolist())


    def astype(
            self,
            dtype: DataType
    ):
        self.data = self.data.astype(dtype.value)
        self.data_type = dtype

        return self


    def detach(self):
        return Tensor(
            data=copy.deepcopy(self.data),
            dtype=self.data_type,
            requires_grad=False
        )


    def clone(self):
        return Tensor(
            data=copy.deepcopy(self.data),
            dtype=self.data_type,
            requires_grad=GradientConfig.enable_grad and self.requires_grad
        )


    # overload atomic-like operations
    def __add__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        """
        Add operation: y = x + other where other is a Tensor or constant

        inputs <- (x, other)
        gradient <- d (Loss) / d (y)

        d (x + other) / d (x) = 1, d (x + other) / d (other) = 1
        gradient -> [x + other] -> (gradient * 1, gradient * 1)
        """
        if isinstance(other, Tensor):
            def _grad_fn_all(gradient, *inputs):
                """
                Taking broadcasting into account:
                    scenario 1: expanding shape of either Tensor with dimensions
                                in the beginning to match shapes
                    scenario 2: expanding dims is not required since shape of Tensor
                                already has 1s and is equal to the shape of the other Tensor 
                """
                # expanded dims
                expanded_dims = tuple((gradient.ndim - tensor.ndim) for tensor in inputs)
                input_gradients = [gradient, gradient]

                for idx, (grad_i, expanded_dims_i) in enumerate(zip(input_gradients, expanded_dims)):
                    # scenario 1: for each dim in the expanded shape, sum out
                    # that dimension since broadcasting was applied to this dim
                    for _ in range(expanded_dims_i):
                        grad_i = config.backend_op('sum', grad_i, axis=0, keepdims=False)

                    # scenario 2: if the shape of the Tensor already has 1s
                    # (without expanding dims), add the weights from that dimension
                    # since broadcasting was applied to this dim. The dimensions
                    # are kept intact since the dimensions were not expanded
                    for i, dim in enumerate(inputs[idx].shape):
                        if dim == 1:
                            grad_i = config.backend_op('sum', grad_i, axis=i, keepdims=True)

                    input_gradients[idx] = grad_i

                return tuple(input_gradients)

            _operation = lambda x, y: config.backend_op('add', x, y)

            if GradientConfig.enable_grad and (self.requires_grad or other.requires_grad):
                node = Operation(
                    name='add',
                    operation=_operation,
                    inputs=(self, other),
                    grad_fn=_grad_fn_all
                )

                result = Tensor(
                    dtype=self.data_type,
                    requires_grad=True
                )

                # result of node
                result.operation = node

            else:
                result = Tensor(
                    data=_operation(self.data, other.data),
                    dtype=self.data_type,
                    requires_grad=False
                )

        else:
            def _grad_fn_one(gradient, *inputs):
                data = inputs[0]
                expanded_dims = gradient.ndim - data.ndim
                input_gradient = gradient

                # taking broadcasting into account
                for _ in range(expanded_dims):
                    input_gradient = config.backend_op('sum', input_gradient, axis=0, keepdims=False)

                for d, dim in enumerate(inputs[0].shape):
                    if dim == 1:
                        input_gradient = config.backend_op('sum', input_gradient, axis=d, keepdims=True)

                return (input_gradient,)

            _operation = lambda x: config.backend_op('add', x, other)

            if GradientConfig.enable_grad and self.requires_grad:
                node = Operation(
                    name='add',
                    operation=_operation,
                    inputs=(self,),
                    grad_fn=_grad_fn_one
                )

                result = Tensor(
                    dtype=self.data_type,
                    requires_grad=True
                )

                result.operation = node

            else:
                result = Tensor(
                    data=_operation(self.data),
                    dtype=self.data_type,
                    requires_grad=False
                )

        return result


    def __sub__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        """
        Subtract operation: y = x - other where other is a Tensor or constant

        inputs <- (x, other)
        gradient <- d (Loss) / d (y)

        d (x - other) / d (x) = 1, d (x - other) / d (other) = -1
        gradient -> [x - other] -> (gradient * 1, gradient * -1)
        """
        if isinstance(other, Tensor):
            return self.__add__(-other)

        else:
            return self.__add__(config.backend_op('negative', other))


    def __mul__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        """
        Element-wise Multiply operation: y = x * other
        where other is a Tensor or constant

        inputs <- (x, other)
        gradient <- d (Loss) / d (y)

        d (x * other) / d (x) = other, d (x * other) / d (other) = x
        gradient -> [x * other] -> (gradient * other, gradient * x)
        """
        if isinstance(other, Tensor):
            def _grad_fn_all(gradient, *inputs):
                # expanded dims
                expanded_dims = tuple((gradient.ndim - tensor.ndim) for tensor in inputs)
                input_gradients = [config.backend_op('multiply', gradient, inputs[1]),
                                   config.backend_op('multiply', gradient, inputs[0])]

                # taking broadcasting into account
                for idx, (grad_i, expanded_dims_i) in enumerate(zip(input_gradients, expanded_dims)):
                    for _ in range(expanded_dims_i):
                        grad_i = config.backend_op('sum', grad_i, axis=0, keepdims=False)

                    for d, dim in enumerate(inputs[idx].shape):
                        if dim == 1:
                            grad_i = config.backend_op('sum', grad_i, axis=d, keepdims=True)

                    input_gradients[idx] = grad_i

                return tuple(input_gradients)

            _operation = lambda x, y: config.backend_op('multiply', x, y)

            if GradientConfig.enable_grad and (self.requires_grad or other.requires_grad):
                node = Operation(
                    name='multiply',
                    operation=_operation,
                    inputs=(self, other),
                    grad_fn=_grad_fn_all
                )

                result = Tensor(
                    dtype=self.data_type,
                    requires_grad=True
                )

                # result of node
                result.operation = node

            else:
                result = Tensor(
                    data=_operation(self.data, other.data),
                    dtype=self.data_type,
                    requires_grad=False
                )

        else:
            def _grad_fn_one(gradient, *inputs):
                data = inputs[0]
                expanded_dims = gradient.ndim - data.ndim
                input_gradient = config.backend_op('multiply', gradient, other)

                # taking broadcasting into account
                for _ in range(expanded_dims):
                    input_gradient = config.backend_op('sum', input_gradient, axis=0, keepdims=False)

                for d, dim in enumerate(inputs[0].shape):
                    if dim == 1:
                        input_gradient = config.backend_op('sum', input_gradient, axis=d, keepdims=True)

                return (input_gradient,)

            _operation = lambda x : config.backend_op('multiply', x, other)

            if GradientConfig.enable_grad and self.requires_grad:
                node = Operation(
                    name='multiply',
                    operation=_operation,
                    inputs=(self,),
                    grad_fn=_grad_fn_one
                )

                result = Tensor(
                    dtype=self.data_type,
                    requires_grad=True
                )

                # result of node
                result.operation = node

            else:
                result = Tensor(
                    data=_operation(self.data),
                    dtype=self.data_type,
                    requires_grad=False
                )
        
        return result


    def __truediv__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        """
        Element-wise Division operation: y = x / other
        where other is a Tensor or constant

        inputs <- (x, other)
        gradient <- d (Loss) / d (y)

        d (x * other) / d (x) = 1 / other,
        d (x * other) / d (other) = -x / (other ** 2)
        gradient -> [x * other] -> (gradient * other, gradient * x)
        """
        if isinstance(other, Tensor):
            def _grad_fn_all(gradient, *inputs):
                expanded_dims = tuple((gradient.ndim - tensor.ndim) for tensor in inputs)
                input_gradients = [config.backend_op('divide', gradient, inputs[1]),
                                   config.backend_op('multiply', config.backend_op('negative', gradient), config.backend_op('divide', inputs[0], config.backend_op('power', inputs[1], 2)))]

                # taking broadcasting into account
                for idx, (grad_i, expanded_dims_i) in enumerate(zip(input_gradients, expanded_dims)):
                    for _ in range(expanded_dims_i):
                        grad_i = config.backend_op('sum', grad_i, axis=0, keepdims=False)

                    for d, dim in enumerate(inputs[idx].shape):
                        if dim == 1:
                            grad_i = config.backend_op('sum', grad_i, axis=d, keepdims=True)

                    input_gradients[idx] = grad_i

                return tuple(input_gradients)

            _operation = lambda x, y: config.backend_op('divide', x, y)

            if GradientConfig.enable_grad and (self.requires_grad or other.requires_grad):
                node = Operation(
                    name='divide',
                    operation=_operation,
                    inputs=(self, other),
                    grad_fn=_grad_fn_all
                )

                result = Tensor(
                    dtype=self.data_type,
                    requires_grad=True
                )

                # result of node
                result.operation = node

            else:
                result = Tensor(
                    data=_operation(self.data, other.data),
                    dtype=self.data_type,
                    requires_grad=False
                )

        else:
            def _grad_fn_one(gradient, *inputs):
                data = inputs[0]
                expanded_dims = gradient.ndim - data.ndim
                input_gradient = config.backend_op('divide', gradient, other)

                # taking broadcasting into account
                for _ in range(expanded_dims):
                    input_gradient = config.backend_op('sum', input_gradient, axis=0, keepdims=False)

                for d, dim in enumerate(inputs[0].shape):
                    if dim == 1:
                        input_gradient = config.backend_op('sum', input_gradient, axis=d, keepdims=True)

                return (input_gradient,)

            _operation = lambda x : config.backend_op('divide', x, other)

            if GradientConfig.enable_grad and self.requires_grad:
                node = Operation(
                    name='divide',
                    operation=_operation,
                    inputs=(self,),
                    grad_fn=_grad_fn_one
                )

                result = Tensor(
                    dtype=self.data_type,
                    requires_grad=True
                )

                # result of node
                result.operation = node

            else:
                result = Tensor(
                    data=_operation(self.data),
                    dtype=self.data_type,
                    requires_grad=False
                )

        return result


    def __neg__(self):
        """
        Negate operation: y = -x

        inputs <- (x,)
        gradient <- d (Loss) / d (y)

        d (-x) / d (x) = -1
        gradient -> [-x] -> (gradient * -1,)
        """
        _operation = lambda x : config.backend_op('negative', x)

        if GradientConfig.enable_grad and self.requires_grad:
            node = Operation(
                name='negative',
                operation=_operation,
                inputs=(self,),
                grad_fn=lambda gradient, *inputs: (config.backend_op('negative', gradient),)
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=True
            )

            # result of node
            result.operation = node

        else:
            result = Tensor(
                data=_operation(self.data),
                dtype=self.data_type,
                requires_grad=False
            )

        return result


    def log(self):
        """
        Element-wise natural Log operation: y = ln(x)

        inputs <- (x,)
        gradient <- d (Loss) / d (y)

        d (ln(x)) / d (x) = 1 / (x * ln(e))
        gradient -> [ln(x)] -> (gradient / (x * ln(e)),)
        """
        _operation = lambda x: config.backend_op('log', x)

        if GradientConfig.enable_grad and self.requires_grad:
            node = Operation(
                name='log',
                operation=_operation,
                inputs=(self,),
                grad_fn=lambda gradient, *inputs: (config.backend_op('divide', gradient, config.backend_op('multiply', inputs[0], config.backend_op('log', math.e))),)
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=True
            )

            # result of node
            result.operation = node

        else:
            result = Tensor(
                data=_operation(self.data),
                dtype=self.data_type,
                requires_grad=False
            )

        return result


    def __pow__(
            self,
            other: Union[int, float, Self, ArrayType]
    ):
        """
        Element-wise Power operation: y = x ** y
        where other is a Tensor or constant

        inputs -> (x, other)
        gradient <- d (Loss) / d (y)

        d (x ^ other) / d (x) = other * (x ^ (other - 1)),
        d (x ^ other) / d (other) = (x ^ other ) * ln(x)
        gradient -> [x ^ other] -> (gradient * other (x ^ (other - 1)),
                                    gradient * (x ^ other) * ln(x))
        """
        if isinstance(other, Tensor):
            def _grad_fn_all(gradient, *inputs):
                # expanded dims
                expanded_dims = tuple((gradient.ndim - tensor.ndim) for tensor in inputs)
                input_gradients = [config.backend_op('multiply', gradient, config.backend_op('multiply', inputs[1], config.backend_op('power', inputs[0], config.backend_op('subtract', inputs[1], 1)))),
                                   config.backend_op('multiply', gradient, config.backend_op('multiply', config.backend_op('power', inputs[0], inputs[1]), config.backend_op('log', inputs[0])))]

                # taking broadcasting into account
                for idx, (grad_i, expanded_dims_i) in enumerate(zip(input_gradients, expanded_dims)):
                    for _ in range(expanded_dims_i):
                        grad_i = config.backend_op('sum', grad_i, axis=0, keepdims=False)

                    for d, dim in enumerate(inputs[idx].shape):
                        if dim == 1:
                            grad_i = config.backend_op('sum', grad_i, axis=d, keepdims=True)

                    input_gradients[idx] = grad_i

                return tuple(input_gradients)

            _operation = lambda x, y: config.backend_op('power', x, y)

            if GradientConfig.enable_grad and (self.requires_grad or other.requires_grad):
                node = Operation(
                    name='power',
                    operation=_operation,
                    inputs=(self, other),
                    grad_fn=_grad_fn_all
                )

                result = Tensor(
                    dtype=self.data_type,
                    requires_grad=True
                )

                # result of node
                result.operation = node

            else:
                result = Tensor(
                    data=_operation(self.data, other.data),
                    dtype=self.data_type,
                    requires_grad=False
                )

        else:
            def _grad_fn_one(gradient, *inputs):
                data = inputs[0]
                expanded_dims = gradient.ndim - data.ndim
                input_gradient = config.backend_op('multiply', gradient, config.backend_op('multiply', other, config.backend_op('power', inputs[0], config.backend_op('subtract', other, 1))))

                # taking broadcasting into account
                for _ in range(expanded_dims):
                    input_gradient = config.backend_op('sum', input_gradient, axis=0, keepdims=False)

                for d, dim in enumerate(inputs[0].shape):
                    if dim == 1:
                        input_gradient = config.backend_op('sum', input_gradient, axis=d, keepdims=True)

                return (input_gradient,)

            _operation = lambda x: config.backend_op('power', x, other)

            if GradientConfig.enable_grad and self.requires_grad:
                node = Operation(
                    name='power',
                    operation=_operation,
                    inputs=(self,),
                    grad_fn=_grad_fn_one
                )

                result = Tensor(
                    dtype=self.data_type,
                    requires_grad=True
                )

                # result of node
                result.operation = node

            else:
                result = Tensor(
                    data=_operation(self.data),
                    dtype=self.data_type,
                    requires_grad=False
                )

        return result


    def sqrt(self):
        """
        Element-wise Square Root operation

        Power operation with other = 0.5
        """
        return self.__pow__(0.5)


    def exp(self):
        """
        Element-wise Exponentiation operation

        Right-hand Power operation with other = e
        """
        return self.__rpow__(math.e)


    # overload right-hand atomic-like operations
    def __iadd__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        if isinstance(other, Tensor):
            if GradientConfig.enable_grad and self.requires_grad:
                raise ValueError("Not supported. Use __add__ instead")

            else:
                self.data = config.backend_op('add', self.data, other.data)

        else:
            if GradientConfig.enable_grad and self.requires_grad:
                raise ValueError("Not supported. Use __add__ instead")

            else:
                self.data = config.backend_op('add', self.data, other)

        return self


    def __radd__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        raise ValueError("Not supported. Use __add__ instead")


    def __isub__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        if isinstance(other, Tensor):
            if GradientConfig.enable_grad and self.requires_grad:
                raise ValueError("Not supported. Use __sub__ instead")

            else:
                self.data = config.backend_op('subtract', self.data, other.data)

        else:
            if GradientConfig.enable_grad and self.requires_grad:
                raise ValueError("Not supported. Use __sub__ instead")

            else:
                self.data = config.backend_op('subtract', self.data, other)

        return self


    def __rsub__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        raise ValueError("Not supported. Use __sub__ instead")


    def __imul__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        if isinstance(other, Tensor):
            if GradientConfig.enable_grad and self.requires_grad:
                raise ValueError("Not supported. Use __mul__ instead")

            else:
                self.data = config.backend_op('multiply', self.data, other.data)

        else:
            if GradientConfig.enable_grad and self.requires_grad:
                raise ValueError("Not supported. Use __mul__ instead")

            else:
                self.data = config.backend_op('multiply', self.data, other)

        return self


    def __rmul__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        raise ValueError("Not supported. Use __mul__ instead")


    def __itruediv__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        if isinstance(other, Tensor):
            if GradientConfig.enable_grad and self.requires_grad:
                raise ValueError("Not supported. Use __truediv__ instead")

            else:
                self.data = config.backend_op('divide', self.data, other.data)

        else:
            if GradientConfig.enable_grad and self.requires_grad:
                raise ValueError("Not supported. Use __truediv__ instead")

            else:
                self.data = config.backend_op('divide', self.data, other)

        return self


    def __rtruediv__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        raise ValueError("Not supported. Use __truediv__ instead")


    def __rpow__(
            self,
            other: Union[int, float, ArrayType]
    ):
        if isinstance(other, Tensor):
            raise ValueError("Not supported. Use __pow__ instead")

        _operation = lambda x: config.backend_op('power', other, x)

        if GradientConfig.enable_grad and self.requires_grad:
            node = Operation(
                name='power',
                operation=_operation,
                inputs=(self,),
                grad_fn=lambda gradient, *inputs: (config.backend_op('multiply', gradient, config.backend_op('multiply', config.backend_op('power', other, inputs[0]), config.backend_op('log', other))),)
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=True
            )

            # result of node
            result.operation = node

        else:
            result = Tensor(
                data=_operation(self.data),
                dtype=self.data_type,
                requires_grad=False
            )

        return result


    # non-atomic operations
    def sum(
            self,
            dim: Optional[Union[int, tuple]] = None,
            keepdim: Optional[bool] = False
    ):
        def _grad_fn(gradient, *inputs):
            if dim is not None and not keepdim:
                gradient = config.backend_op('expand_dims', gradient, axis=dim)

            input_gradient = config.backend_op('multiply', gradient, config.backend_op('ones_like', inputs[0]))

            return (input_gradient,)

        _operation = lambda x: config.backend_op('sum', x, axis=dim, keepdims=keepdim)

        if GradientConfig.enable_grad and self.requires_grad:
            node = Operation(
                name='sum',
                operation=_operation,
                inputs=(self,),
                grad_fn=_grad_fn
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=True
            )

            result.operation = node

        else:
            result = Tensor(
                data=_operation(self.data),
                dtype=self.data_type,
                requires_grad=False
            )

        return result


    def mean(
            self,
            dim: Optional[Union[int, tuple]] = None,
            keepdim: Optional[bool] = False
    ):
        def _grad_fn(gradient, *inputs):
            if dim is not None:
                if not keepdim:
                    gradient = config.backend_op('expand_dims', gradient, axis=dim)

                shape_prod = prod(tuple(self.shape[d] for d in dim)) if isinstance(dim, (tuple, list)) else self.shape[dim]

            else:
                shape_prod = prod(self.shape)

            input_gradient = config.backend_op('multiply', gradient, config.backend_op('divide', config.backend_op('ones_like', inputs[0]), shape_prod))

            return (input_gradient,)

        _operation = lambda x: config.backend_op('mean', x, axis=dim, keepdims=keepdim)


        if GradientConfig.enable_grad and self.requires_grad:
            node = Operation(
                name='mean',
                operation=_operation,
                inputs=(self,),
                grad_fn=_grad_fn
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=True
            )

            result.operation = node

        else:
            result = Tensor(
                data=_operation(self.data),
                dtype=self.data_type,
                requires_grad=False
            )

        return result


    def max(
            self,
            dim: Optional[Union[int, tuple]] = None,
            keepdim: Optional[bool] = False
    ):
        def _grad_fn(gradient, *inputs):
            data = inputs[0]
            input_gradient = config.backend_op('zeros_like', data)

            if dim is not None and not keepdim:
                gradient = config.backend_op('expand_dims', gradient, axis=dim)

            max_values = config.backend_op('max', data, axis=dim, keepdims=True)

            # propogate the gradient only through the maximum values
            # of the result tensor
            input_gradient = config.backend_op('where', config.backend_op('equal', data, max_values),
                                               gradient, 0)

            return (input_gradient,)

        _operation = lambda x: config.backend_op('max', x, axis=dim, keepdims=keepdim)


        if GradientConfig.enable_grad and self.requires_grad:
            node = Operation(
                name='max',
                operation=_operation,
                inputs=(self,),
                grad_fn=_grad_fn
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=True
            )

            result.operation = node

        else:
            result = Tensor(
                data=_operation(self.data),
                dtype=self.data_type,
                requires_grad=False
            )

        return result


    def matmul(
            self,
            other: Self
    ):
        """
        MatMul operation: y = x @ other
        where other is a Tensor

        inputs <- (x, other)
        gradient <- d (Loss) / d (y)

        Currently, grad_fn works when x is batched: shape: (3,) => (1, 3)

        d (x @ other) / d (x) = ,
        d (x @ other) / d (other) = 
        gradient -> [x @ other] -> (gradient @ other.T, x.T @ gradient)
        """
        if not isinstance(other, Tensor):
            raise ValueError("MatMul is only supported when other is a Tensor")

        _operation = lambda x, y: config.backend_op('matmul', x, y)

        if GradientConfig.enable_grad and (self.requires_grad or other.requires_grad):
            node = Operation(
                name='matmul',
                operation=_operation,
                inputs=(self, other),
                grad_fn=lambda gradient, *inputs: (config.backend_op('matmul', gradient, config.backend_op('transpose', inputs[1])),
                                                   config.backend_op('matmul', config.backend_op('transpose', inputs[0]), gradient))
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=True
            )

            # result of node
            result.operation = node

        else:
            result = Tensor(
                data=_operation(self.data, other.data),
                dtype=self.data_type,
                requires_grad=False
            )
    
        return result


    # transformation operations
    def set_slice(
            self,
            indices,
            value: Self
    ):
        """
        Support Tensor indexing

        Set slices of elements from Tensor to value

        out[..., 1:3] = x is equivalent to
        out = out.set_slice((Ellipsis, slice(1, 3)), x)
        """
        def _grad_fn(gradient, *inputs):
            input_gradient = config.backend_op('zeros_like', inputs[0])
            input_gradient[indices] = gradient

            return (input_gradient,)

        def _operation(x, y):
            x[indices] = y

            return x


        if GradientConfig.enable_grad and self.requires_grad:
            node = Operation(
                name='set_slice',
                operation=_operation,
                inputs=(self, value),
                grad_fn=_grad_fn
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=True
            )

            # result of node
            result.operation = node

        else:
            result = Tensor(
                data=_operation(self.data, value.data),
                dtype=self.data_type,
                requires_grad=False
            )

        return result


    def transpose(
            self,
            dim: Optional[tuple] = None
    ):
        """
        Transpose operation at the specified axes: y = x.T

        inputs <- (x,)
        gradient <- d (Loss) / d (y)

        d (x.T) / d (x) = 1.T
        gradient -> [x.T] -> (gradient.T,)
        """
        _operation = lambda x: config.backend_op('transpose', x, axes=dim)

        if GradientConfig.enable_grad and self.requires_grad:
            node = Operation(
                name='transpose',
                operation=_operation,
                inputs=(self,),
                grad_fn=lambda gradient, *inputs: (config.backend_op('transpose', gradient, axes=dim),)
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=True
            )

            result.operation = node

        else:
            result = Tensor(
                data=_operation(self.data),
                dtype=self.data_type,
                requires_grad=False
            )

        return result


    def reshape(
            self,
            shape: tuple
    ):
        """
        Return a new Tensor which is the reshape of self: y = x.reshape(shape)
        Currently, this operation fires the computational graph

        inputs <- (x,)
        gradient <- d (Loss) / d (y) <- d (Loss) / d (reshape(x, shape))

        In order to propogate the gradient back to x, gradient needs to be reshaped
        gradient -> [y] -> (x.reshape(original_shape),)
        """
        _operation = lambda x: config.backend_op('reshape', x, shape)

        if GradientConfig.enable_grad and self.requires_grad:
            node = Operation(
                name='reshape',
                operation=_operation,
                inputs=(self,),
                grad_fn=lambda gradient, *inputs: (config.backend_op('reshape', gradient, self.shape),)
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=True
            )

            result.operation = node

        else:
            result = Tensor(
                data=_operation(self.data),
                dtype=self.data_type,
                requires_grad=False
            )

        return result


    def squeeze(
            self,
            dim: Optional[Union[int, tuple]] = ()
    ):
        _operation = lambda x: config.backend_op('squeeze', x, axis=dim)

        if GradientConfig.enable_grad and self.requires_grad:
            node = Operation(
                name='squeeze',
                operation=_operation,
                inputs=(self,),
                grad_fn=lambda gradient, *inputs: config.backend_op('expand_dims', gradient, axis=dim)
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=True
            )

            result.operation = node

        else:
            result = Tensor(
                data=_operation(self.data),
                dtype=self.data_type,
                requires_grad=False
            )

        return result


    def unsqueeze(
            self,
            dim: Optional[Union[int, tuple]] = ()
    ):
        _operation = lambda x: config.backend_op('expand_dims', x, axis=dim)

        if GradientConfig.enable_grad and self.requires_grad:
            node = Operation(
                name='unsqueeze',
                operation=_operation,
                inputs=(self,),
                grad_fn=lambda gradient, *inputs: config.backend_op('squeeze', gradient, axis=dim)
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=True
            )

            result.operation = node

        else:
            result = Tensor(
                data=_operation(self.data),
                dtype=self.data_type,
                requires_grad=False
            )

        return result


    def cat(
            self,
            tensors: tuple,
            dim: int = 0
    ):
        def _grad_fn(gradient, *inputs):
            input_split_size = tuple(node_input.shape[dim] for node_input in inputs)
            input_split_size = tuple(config.backend_op('cumsum', config.Array(input_split_size[:-1])).tolist())

            return tuple(config.backend_op('split', gradient, indices_or_sections=input_split_size, axis=dim))

        _operation = lambda *inputs: config.backend_op('concatenate', inputs, axis=dim)


        if GradientConfig.enable_grad and self.requires_grad:
            node = Operation(
                name='cat',
                operation=_operation,
                inputs=(self,) + tensors,
                grad_fn=_grad_fn
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=True
            )

            result.operation = node

        else:
            result = Tensor(
                data=_operation(self.data),
                dtype=self.data_type,
                requires_grad=False
            )

        return result


    def split(
            self,
            split_size_or_sections: Union[int, tuple],
            dim: int = 0
    ):
        _operation = lambda x: config.backend_op('split', x, indices_or_sections=split_size_or_sections, axis=dim)

        split_sizes = split_size_or_sections if isinstance(split_size_or_sections, int) else len(split_size_or_sections)

        result = ()

        if GradientConfig.enable_grad or self.requires_grad:
            node = Operation(
                name='split',
                operation=_operation,
                inputs=(self,),
                grad_fn=lambda gradient, *inputs: config.backend_op('concatenate', gradient, axis=dim)
            )

            for idx in range(split_sizes):
                r = Tensor(
                    dtype=self.data_type,
                    requires_grad=True
                )

                r.operation = node
                r.output_index = idx

                result += (r,)

        else:
            computed_result = _operation(self.data)

            for idx in range(split_sizes):
                r = Tensor(
                    data=computed_result[idx],
                    dtype=self.data_type,
                    requires_grad=False
                )

                result += (r,)

        return result


    def repeat_interleave(
            self,
            repeats: int,
            dim: Optional[int] = None
    ):
        def _grad_fn(gradient, *inputs):
            # if no dim is None, the tensor is flattened, so sum
            # over the repeated elements. reshape(gradient, (-1, repeats))
            # puts the repeated elements of each element in axis=1
            if dim is None:
                input_gradient = config.backend_op('sum', config.backend_op('reshape', gradient, (-1, repeats)), axis=1)
                input_gradient = config.backend_op('reshape', input_gradient, inputs[0].shape)

            else:
                grad_ndim = gradient.ndim
                p_dim = dim if dim >= 0 else grad_ndim + dim # account for negative dim

                input_gradient = config.backend_op('zeros_like', inputs[0])

                # split the gradient into chunks of repeated elements along axis=dim
                gradient_split = config.backend_op('split', gradient, config.backend_op('cumsum', config.Array((repeats,) * inputs[0].shape[p_dim]))[:-1], axis=p_dim)

                # and add the repeated elements in gradient and copy them
                # over to the input gradient at the appropriate indices
                for i, grad_split_i in enumerate(gradient_split):
                    # this is equivalent to Array[..., dim, ...]
                    indices = [slice(None)] * grad_ndim
                    indices[p_dim] = i

                    input_gradient[tuple(indices)] = config.backend_op('sum', grad_split_i, axis=p_dim, keepdims=False)

            return (input_gradient,)

        _operation = lambda x: config.backend_op('repeat', x, repeats, axis=dim)


        if GradientConfig.enable_grad and self.requires_grad:
            node = Operation(
                name='repeat_interleave',
                operation=_operation,
                inputs=(self,),
                grad_fn=_grad_fn
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=True
            )

            result.operation = node

        else:
            result = Tensor(
                data=_operation(self.data),
                dtype=self.data_type,
                requires_grad=False
            )

        return result


    def as_strided(
            self,
            shape,
            strides
    ):
        _operation = lambda x: config.strided_lib.as_strided(x, shape, strides)

        if GradientConfig.enable_grad and self.requires_grad:
            node = Operation(
                name='reshape',
                operation=_operation,
                inputs=(self,),
                grad_fn=lambda gradient, *inputs: (config.strided_lib.as_strided(gradient, self.shape, self.stride),)
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=True
            )

            result.operation = node

        else:
            result = Tensor(
                data=_operation(self.data),
                dtype=self.data_type,
                requires_grad=False
            )

        return result


    def maximum(
            self,
            other: Union[int, float, Self]
    ):
        """
        Maxmimum operation between self and other
        
        y = maximum(x, other) => y = x where x >= other else other

        inputs <- (x,)
        gradient <- d (Loss) / d (y)

        d (y) / d (x) = d (maximum(x - other, 0) / d (x) = sign(maximum(x - other), 0)
        d (y) / d (other) = d (maximum(x - other, 0) / d (other) = sign(maximum(other - x), 0)
        gradient -> [x.T] -> (gradient * (d (y) / d (x)), gradient * (d (y) / d (other)))
        """
        if isinstance(other, Tensor):
            _operation = lambda x, y: config.backend_op('maximum', x, y)

            if GradientConfig.enable_grad and (self.requires_grad or other.requires_grad):
                node = Operation(
                    name='maximum',
                    operation=_operation,
                    inputs=(self, other),
                    grad_fn=lambda gradient, *inputs: (config.backend_op('multiply', gradient, config.backend_op('sign', config.backend_op('maximum', config.backend_op('subtract', inputs[0], inputs[1]), 0))),
                                                       config.backend_op('multiply', gradient, config.backend_op('sign', config.backend_op('maximum', config.backend_op('subtract', inputs[1], inputs[0]), 0))))
                )

                result = Tensor(
                    dtype=self.data_type,
                    requires_grad=True
                )

                result.operation = node

            else:
                result = Tensor(
                    data=_operation(self.data, other.data),
                    dtype=self.data_type,
                    requires_grad=False
                )

        else:
            _operation = lambda x: config.backend_op('maximum', x, other)

            if GradientConfig.enable_grad and self.requires_grad:
                node = Operation(
                    name='maximum',
                    operation=_operation,
                    inputs=(self,),
                    grad_fn=lambda gradient, *inputs: (config.backend_op('multiply', gradient, config.backend_op('sign', config.backend_op('maximum', config.backend_op('subtract', inputs[0], other), 0))),)
                )

                result = Tensor(
                    dtype=self.data_type,
                    requires_grad=True
                )

                result.operation = node

            else:
                result = Tensor(
                    data=_operation(self.data),
                    dtype=self.data_type,
                    requires_grad=False
                )

        return result


    def einsum(
            self,
            subscripts: str,
            *tensors: Self
    ):
        def _reorder_subscripts(ordered_subscripts, output_subscript, index):
            main_tensor_subscript = ordered_subscripts[index]
            other_tensor_subscripts = tuple(ordered_subscripts[idx] for idx in range(len(ordered_subscripts)) if idx != index)

            # other_tensor_subscripts is empty when einsum operation
            # is performed only on self with no other tensors
            if other_tensor_subscripts:
                return "{},{}->{}".format(output_subscript, ','.join(other_tensor_subscripts), main_tensor_subscript)

            else:
                return "{}->{}".format(output_subscript, main_tensor_subscript)

        def _grad_fn(gradient, *inputs):
            # subscripts split for each input/tensor, in same order as *tensors
            input_subscript, output_subscript = subscripts.split('->')
            ordered_subscripts = tuple(input_subscript.split(','))

            gradients = ()

            for i in range(len(inputs)):
                # find dimensions (characters) in inputs and output (subscripts)
                # that are not present in the output and are not repeated in the
                # inputs which means that these dimension have been reduced/summed over
                replace_dims = tuple((shape[dim], char) for (shape, subscript) in zip((_input.shape for _input in inputs), ordered_subscripts) for dim, char in enumerate(subscript) if char not in output_subscript)
                recoup_dims = tuple(replace_dim for replace_dim, count in collections.Counter(replace_dims).items() if count == 1)

                if recoup_dims:
                    gradient_shape = gradient.shape

                    shape = tuple(recoup_dim[0] for recoup_dim in recoup_dims)
                    recoup_subscript = ''.join(tuple(recoup_dim[1] for recoup_dim in recoup_dims))

                    output_subscript = output_subscript + recoup_subscript

                    # if the input tensor had been reduced over its axes,
                    # expand and broadcast the gradient to get back the original shape
                    expanded_shape = gradient_shape + shape
                    gradient = config.backend_op('broadcast_to', config.backend_op('expand_dims', gradient, axis=tuple(range(len(gradient_shape), len(expanded_shape)))), expanded_shape)

                reordered_subscripts = _reorder_subscripts(ordered_subscripts, output_subscript, i)

                # reverse the einsum computation on the incoming gradient
                # to get the local gradients
                gradients += (config.backend_op('einsum', reordered_subscripts, gradient, *(_input for j, _input in enumerate(inputs) if j != i)),)

            return gradients

        _operation = lambda *tensors: config.backend_op('einsum', subscripts, *tensors)


        if not all(isinstance(tensor, Tensor) for tensor in tensors):
            raise ValueError("Einsum is only supported when 'tensors' argument are Tensors")

        if GradientConfig.enable_grad and (self.requires_grad or any(tensor.requires_grad for tensor in tensors)):
            node = Operation(
                name='einsum',
                operation=_operation,
                inputs=(self,) + tensors,
                grad_fn=_grad_fn
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=True
            )

            result.operation = node

        else:
            result = Tensor(
                data=_operation(self.data, *(tensor.data for tensor in tensors)),
                dtype=self.data_type,
                requires_grad=False
            )

        return result


    def zero_grad(
            self
    ):
        if self.requires_grad:
            self.grad = zeros_like(self.data, requires_grad=False)

        else:
            self.grad = None


    def backward(
            self,
            gradient: Optional[Self] = None
    ):
        if (not GradientConfig.enable_grad) or (not self.requires_grad):
            return

        # grad: gradient of a loss with respect to the node's output 
        # is initialtized to 1: to start with computing
        # gradient of loss with respect to itself
        if gradient is None:
            gradient = ones_like(self.data, requires_grad=False)

        # accumulate gradient at the current node: chain rule
        # config.backend_op('add', self.grad, gradient) # implicit broadcasting
        self.grad += gradient # in-place

        if self.operation is not None:
            self.operation.backward(self.grad)


    def sprint(self):
        return "Tensor(shape: {}, dtype: {}, requires_grad: {})".format(
            self.data.shape, self.data_type.value_as_str, self.requires_grad
        )


    def __repr__(self):
        return "Tensor(data: {}, shape: {}, dtype: {}, grad: {}, requires_grad: {})".format(
            self.data, self.data.shape, self.data_type.value_as_str, self.grad, self.requires_grad
        )


@jaxtyped(typechecker=typechecker)
class Operation:
    """
    Class that represents a single node in the computational graph

    Each node is an operation in the computational graph

    Incorporates lazy computation and backpropagation
    """
    def __init__(
            self,
            name: str,
            operation: Callable,
            inputs: tuple,
            grad_fn: Union[Callable, None] = None
    ):
        # value of node for caching
        self.value: Union[None, tuple, list, ArrayType] = None

        # operation fn
        self.operation = operation

        # operation name
        self.name = name if name is not None else ''

        # inputs to the node
        self.inputs = tuple(inputs)

        # function to compute gradient of the node's output
        # with respect to each of the node's inputs
        self.grad_fn = grad_fn

        # does value need to be evaluated: lazy computation
        self.fired = False


    def fire(self):
        """
        Evaluate value of node by applying operation
        to its input when fire() is called

        If value has already been evaluated, return value
        Else, compute the value and save it
        """
        # if self.fired = False, perform computation
        if not self.fired:
            # update self.inputs with the computed Tensor
            # after computing the inputs to the Operation
            # self.inputs = tuple(node_input.fire() for node_input in self.inputs)
            for node_input in self.inputs:
                fire(node_input)

            # compute the values of the inputs to the operation
            # and generate a new Tensor
            self.value = self.operation(*(tensor.data for tensor in self.inputs))

            self.fired = True

        return self.value


    def backward(
            self,
            gradient: Union[tuple, Tensor]
    ):
        """
        Implement backpropogation: compute gradients for node
        in the computational graph
        """
        if isinstance(gradient, (tuple, list)):
            gradient = tuple(g.data for g in gradient)

        else:
            gradient = gradient.data

        # compute gradients with respect to the node inputs
        # and propogate the gradients backward through the graph
        # by recursively calling backward() on the input nodes
        if self.grad_fn is not None:
            gradients: tuple = self.grad_fn(gradient,
                                            *(input_tensor.data for input_tensor in self.inputs))

            for node_input, input_grad in zip(self.inputs, gradients):
                node_input.backward(Tensor(input_grad, requires_grad=False))


    def sprint(self):
        return "Operation(op: {}, value_shape: {}, fired: {})".format(
            self.name,
            None if self.value is None else tuple(v.shape for v in self.value) if isinstance(self.value, (list, tuple)) else self.value.shape,
            self.fired
        )


    def __repr__(self):
        return "Operation(op: {}, value: {}, fired: {})".format(
            self.name, self.value, self.fired
        )


# functions
# defined here to avoid circular import dependency
@jaxtyped(typechecker=typechecker)
def zeros_like(
        tensor: Union[Tensor, ArrayType],
        requires_grad: Optional[bool] = None
):
    if isinstance(tensor, ArrayType):
        return Tensor(
            data=config.backend_op('zeros_like', tensor),
            dtype=DataType(str(tensor.dtype)),
            requires_grad=requires_grad if requires_grad is not None else False
        )

    else:
        return Tensor(
            data=config.backend_op('zeros_like', tensor.data),
            dtype=tensor.data_type,
            requires_grad=requires_grad if requires_grad is not None else tensor.requires_grad
        )


@jaxtyped(typechecker=typechecker)
def ones_like(
        tensor: Union[Tensor, ArrayType],
        requires_grad: Optional[bool] = None
):
    if isinstance(tensor, ArrayType):
        return Tensor(
            data=config.backend_op('ones_like', tensor),
            dtype=DataType(str(tensor.dtype)),
            requires_grad=requires_grad if requires_grad is not None else False
        )

    else:
        return Tensor(
            data=config.backend_op('ones_like', tensor.data),
            dtype=tensor.data_type,
            requires_grad=requires_grad if requires_grad is not None else tensor.requires_grad
        )


@jaxtyped(typechecker=typechecker)
def full(
        shape: tuple,
        fill_value: Union[int, float],
        dtype: Optional[DataType] = DataType('float32'),
        requires_grad: Optional[bool] = False
):
    return Tensor(
        data=config.backend_op('full', shape, fill_value),
        dtype=dtype,
        requires_grad=requires_grad
    )


@jaxtyped(typechecker=typechecker)
def fire(root: Tensor):
    """
    Fire the computational graph
    """
    operation = root.operation

    if operation is not None:
        # update result: if operation was already fired,
        # operation.fire() return the cached result stored
        # in self.value
        result: Union[tuple, list, ArrayType] = operation.fire()

        if not isinstance(result, ArrayType):
            root.data = result[root.output_index]

        else:
            root.data = result

        # initialize gradient for Tensor
        if root.requires_grad and root.grad.numel() == 0:
            root.zero_grad()

