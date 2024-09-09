from typing import Any, Union, Optional, Self
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

import copy
import collections
import functools

from .config import config, ArrayType


@jaxtyped(typechecker=typechecker)
class Device:
    def __init__(
            self,
            device: str = 'cpu'
    ):
        self.value = config.get_device(device)


@jaxtyped(typechecker=typechecker)
class DataType:
    def __init__(
            self,
            dtype: str = 'float32'
    ):
        self.value_as_str = dtype

        # to accomodate for dtype strings
        # like "mlx.core.float32"
        if 'float32' in dtype.lower():
            self.value = config.backend.float32

        elif 'float16' in dtype.lower():
            self.value = config.backend.float16

        else:
            raise ValueError("dtype is not a valid datatype or is not supported")


@jaxtyped(typechecker=typechecker)
class Tensor:
    """
    Class that represents a Tensor datatype

    Each node is a Tensor in the computational graph

    Incorporates lazy computation and automatic differentiation
    """
    def __init__(
            self,
            data: Any = (),
            dtype: Optional[DataType] = DataType('float32'),
            requires_grad: Optional[bool] = False
    ):
        data = () if data is None else data

        # value of node: can be result of an operation or constant value
        if not isinstance(data, ArrayType):
            self.data = config.Array(data, dtype=dtype.value)

        else:
            self.data = data.astype(dtype=dtype.value)

        # dtype of Tensor
        self.data_type = dtype

        # operation in the computational graph
        # that connects this Tensor
        self.node: Operation = None

        # whether to compute gradient for the node
        self.requires_grad = requires_grad

        # gradient with respect to this node
        self.zero_grad()


    def __array__(self):
        """
        For converting to numpy array,
        using numpy.array()
        """
        return NotImplementedError


    def numel(self) -> int:
        """
        Number of elements in Tensor
        """
        return self.data.size


    def __getitem__(
            self,
            indices
    ):
        """
        Support Tensor indexing

        Retrieve slices of elements from Tensor
        """
        def _grad_fn(gradient: ArrayType, *inputs: Optional[tuple]):
            grad = zeros_like(gradient).data
            grad[indices] = gradient

            return (grad,)


        node = Operation(
            name='get_slice',
            operation=lambda x: x[indices],
            inputs=(self,),
            grad_fn=_grad_fn
        )

        result = Tensor(
            dtype=self.data_type,
            requires_grad=self.requires_grad
        )

        # result of node
        result.node = node

        return result


    def __setitem__(
            self,
            indices,
            value
    ):
        """
        Support Tensor indexing

        Set slices of elements from Tensor to value

        This operation fires the computational graph
        """
        if self.numel() == 0:
            result = self.node.fire()

            self.__dict__.update(result.__dict__)

        self.data[indices] = (value.data if isinstance(value, Tensor)
                              else value)


    @property
    def shape(self) -> tuple:
        if self.numel() == 0:
            # when .shape is called,
            # if Tensor has not been computed,
            # compute the Tensor
            fired_tensor = self.fire()

            # updated the Tensor after computation
            self.__dict__.update(fired_tensor.__dict__)

        return self.data.shape


    @property
    def dtype(self):
        return self.data_type


    @property
    def itemsize(self):
        return self.data.itemsize


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

        return strides.tolist()


    def astype(
            self,
            dtype: DataType
    ):
        self.data = self.data.astype(dtype.value)
        self.data_type = dtype


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
            requires_grad=self.requires_grad
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
            node = Operation(
                name='__add__',
                operation=lambda x, y: config.backend.add(x, y),
                inputs=(self, other),
                grad_fn=lambda gradient, *inputs: (gradient, gradient)
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=(self.requires_grad or other.requires_grad)
            )

            # result of node
            result.node = node

        else:
            node = Operation(
                name='__add__',
                operation=lambda x: config.backend.add(x, other),
                inputs=(self,),
                grad_fn=lambda gradient, *inputs: (gradient,)
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=self.requires_grad
            )

            result.node = node

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
            node = Operation(
                name='__sub__',
                operation=lambda x, y: config.backend.subtract(x, y),
                inputs=(self, other),
                grad_fn=lambda gradient, *inputs: (gradient, -gradient)
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=(self.requires_grad or other.requires_grad)
            )

            # result of node
            result.node = node

        else:
            node = Operation(
                name='__sub__',
                operation=lambda x: config.backend.subtract(x, other),
                inputs=(self,),
                grad_fn=lambda gradient, *inputs: (gradient,)
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=self.requires_grad
            )

            result.node = node

        return result


    def mul(
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
            node = Operation(
                name='mul',
                operation=lambda x, y: x * y,
                inputs=(self, other),
                grad_fn=lambda gradient, *inputs: (gradient * inputs[1],
                                                   gradient * inputs[0])
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=(self.requires_grad or other.requires_grad)
            )

            # result of node
            result.node = node

        else:
            node = Operation(
                name='mul',
                operation=lambda x : x * other,
                inputs=(self,),
                grad_fn=lambda gradient, *inputs: (gradient * inputs[1],)
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=self.requires_grad
            )

            # result of node
            result.node = node
        
        return result


    def div(
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
            node = Operation(
                name='div',
                operation=lambda x, y: x / y,
                inputs=(self, other),
                grad_fn=lambda gradient, *inputs: (gradient / inputs[1],
                                                   -gradient * (inputs[0] / inputs[1] ** 2))
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=(self.requires_grad or other.requires_grad)
            )

            # result of node
            result.node = node

        else:
            node = Operation(
                name='div',
                operation=lambda x : x / other,
                inputs=(self,),
                grad_fn=lambda gradient, *inputs: (gradient / inputs[1],)
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=self.requires_grad
            )

            # result of node
            result.node = node
        
        return result


    def __neg__(self):
        """
        Negate operation: y = -x

        inputs <- (x,)
        gradient <- d (Loss) / d (y)

        d (-x) / d (x) = -1
        gradient -> [-x] -> (gradient * -1,)
        """
        node = Operation(
            name='__neg__',
            operation=lambda x : -x,
            inputs=(self,),
            grad_fn=lambda gradient, *inputs: (-gradient,)
        )

        result = Tensor(
            dtype=self.data_type,
            requires_grad=self.requires_grad
        )

        # result of node
        result.node = node

        return result


    def __exp__(self):
        """
        Element-wise Exponentiation operation: y = e^x

        inputs <- (x,)
        gradient <- d (Loss) / d (y)

        d (e^x) / d (x) = e^x
        gradient -> [e^x] -> (gradient * e^x,)
        """
        node = Operation(
            name='__exp__',
            operation=lambda x: config.backend.exp(x),
            inputs=(self,),
            grad_fn=lambda gradient, *inputs: (gradient * config.backend.exp(inputs[0]),)
        )

        result = Tensor(
            dtype=self.data_type,
            requires_grad=self.requires_grad
        )

        result.node = node

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
            node = Operation(
                name='__power__',
                operation=lambda x, y: config.backend.power(x, y),
                inputs=(self, other),
                grad_fn=lambda gradient, *inputs: (gradient * inputs[1] * config.backend.power(inputs[0], inputs[1] - 1),
                                                   gradient * config.backend.power(inputs[0], inputs[1]) * config.backend.log(inputs[0]))
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=(self.requires_grad or other.requires_grad)
            )

            # result of node
            result.node = node

        else:
            node = Operation(
                name='__power__',
                operation=lambda x: config.backend.power(x, other),
                inputs=(self,),
                grad_fn=lambda gradient, *inputs: (gradient * inputs[1] * config.backend.power(inputs[0], inputs[1] - 1),)
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=self.requires_grad
            )

            result.node = node

        return result


    def sqrt(self):
        """
        Element-wise Square Root operation

        Power operation with other = 0.5
        """
        return self.__pow__(0.5)


    # overload right-hand atomic-like operations
    def __iadd__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        raise ValueError("Not supported. Use __add__ instead")


    def __radd__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        raise ValueError("Not supported. Use __add__ instead")


    def __isub__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        raise ValueError("Not supported. Use __sub__ instead")


    def __rsub__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        raise ValueError("Not supported. Use __sub__ instead")


    def __mul__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        raise ValueError("Not supported. Use .mul() instead")


    def __imul__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        raise ValueError("Not supported. Use .mul() instead")


    def __rmul__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        raise ValueError("Not supported. Use .mul() instead")


    def __truediv__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        raise ValueError("Not supported. Use .div() instead")


    def __itruediv__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        raise ValueError("Not supported. Use .div() instead")


    def __rtruediv__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        raise ValueError("Not supported. Use .div() instead")


    # non-atomic operations
    def sum(
            self,
            dim: Optional[tuple] = (),
            keepdim: Optional[bool] = False
    ):
        raise ValueError("Not supported. Use .einsum() instead")


    def mean(
            self,
            dim: Optional[tuple] = (),
            keepdim: Optional[bool] = False
    ):
        node = Operation(
            name='mean',
            operation=lambda x: config.backend.mean(x, axis=dim, keepdims=keepdim),
            inputs=(self,),
            grad_fn=lambda gradient, *inputs: (config.backend.expand_dims(gradient, axis=dim) *
                                               (config.backend.ones_like(inputs[0]) / prod(tuple(self.shape[d] for d in dim))),)
        )

        result = Tensor(
            dtype=self.data_type,
            requires_grad=self.requires_grad
        )

        result.node = node

        return result


    def max(
            self,
            dim: Optional[tuple] = (),
            keepdim: Optional[bool] = False
    ):
        def _grad_fn(gradient, *inputs):
            data = inputs[0]
            _gradient = config.backend.zeros_like(data)

            max_values = config.backend.max(data, axis=dim, keepdims=True)

            # propogate the gradient only through the maximum values
            # of the result tensor
            _gradient = config.backend.where(data == max_values,
                                             config.backend.expand_dims(gradient, axis=dim),
                                             0)

            return (_gradient,)


        node = Operation(
            name='max',
            operation=lambda x: config.backend.max(x, axis=dim, keepdims=keepdim),
            inputs=(self,),
            grad_fn=_grad_fn
        )

        result = Tensor(
            dtype=self.data_type,
            requires_grad=self.requires_grad
        )

        result.node = node

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

        node = Operation(
            name='matmul',
            operation=lambda x, y: config.backend.matmul(x, y),
            inputs=(self, other),
            grad_fn=lambda gradient, *inputs: (config.backend.matmul(gradient, inputs[1].T),
                                               config.backend.matmul(inputs[0].T, gradient))
        )

        result = Tensor(
            dtype=self.data_type,
            requires_grad=(self.requires_grad or other.requires_grad)
        )

        # result of node
        result.node = node
    
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
        def _grad_fn(gradient, *tensors):
            grad = zeros_like(gradient).data
            grad[indices] = gradient

            return (grad,)

        def _operation(x, y):
            # x[indices] = y.astype(x.dtype)
            x[indices] = y

            return x


        node = Operation(
            name='set_slice',
            operation=_operation,
            inputs=(self, value),
            grad_fn=_grad_fn
        )

        result = Tensor(
            dtype=self.data_type,
            requires_grad=self.requires_grad
        )

        # result of node
        result.node = node

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
        node = Operation(
            name='transpose',
            operation=lambda x: config.backend.transpose(x, dim),
            inputs=(self,),
            grad_fn=lambda gradient, *inputs: (config.backend.transpose(gradient, dim),)
        )

        result = Tensor(
            dtype=self.data_type,
            requires_grad=self.requires_grad
        )

        result.node = node

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
        node = Operation(
            name='reshape',
            operation=lambda x: config.backend.reshape(x, shape),
            inputs=(self,),
            grad_fn=lambda gradient, *inputs: (config.backend.reshape(gradient, self.shape),)
        )

        result = Tensor(
            dtype=self.data_type,
            requires_grad=self.requires_grad
        )

        result.node = node

        return result


    def squeeze(
            self,
            dim: Optional[Union[int, tuple]] = ()
    ):
        node = Operation(
            name='squeeze',
            operation=lambda x: config.backend.squeeze(x, axis=dim),
            inputs=(self,),
            grad_fn=lambda gradient, *inputs: config.backend.expand_dims(gradient, axis=dim)
        )

        result = Tensor(
            dtype=self.data_type,
            requires_grad=self.requires_grad
        )

        result.node = node

        return result


    def unsqueeze(
            self,
            dim: Optional[Union[int, tuple]] = ()
    ):
        node = Operation(
            name='unsqueeze',
            operation=lambda x: config.backend.expand_dims(x, axis=dim),
            inputs=(self,),
            grad_fn=lambda gradient, *inputs: config.backend.squeeze(gradient, axis=dim)
        )

        result = Tensor(
            dtype=self.data_type,
            requires_grad=self.requires_grad
        )

        result.node = node

        return result


    def as_strided(
            self,
            shape,
            strides
    ):
        node = Operation(
            name='reshape',
            operation=lambda x: config.strided_lib.as_strided(x, shape, strides),
            inputs=(self,),
            grad_fn=lambda gradient, *inputs: (config.strided_lib.as_strided(gradient, self.shape, self.stride),)
        )

        result = Tensor(
            dtype=self.data_type,
            requires_grad=self.requires_grad
        )

        result.node = node

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
            node = Operation(
                name='maximum',
                operation=lambda x, y: config.backend.maximum(x, y),
                inputs=(self, other),
                grad_fn=lambda gradient, *inputs: (gradient * config.backend.sign(config.backend.maximum(inputs[0] - inputs[1], 0)),
                                                   gradient * config.backend.sign(config.backend.maximum(inputs[1] - inputs[0], 0)))
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=(self.requires_grad or other.requires_grad)
            )

            result.node = node

        else:
            node = Operation(
                name='maximum',
                operation=lambda x: config.backend.maximum(x, other),
                inputs=(self,),
                grad_fn=lambda gradient, *inputs: (gradient * config.backend.sign(config.backend.maximum(inputs[0] - other, 0)),)
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=self.requires_grad
            )

            result.node = node

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
                    gradient = config.backend.broadcast_to(config.backend.expand_dims(gradient, axis=tuple(range(len(gradient_shape), len(expanded_shape)))), expanded_shape)

                reordered_subscripts = _reorder_subscripts(ordered_subscripts, output_subscript, i)

                # reverse the einsum computation on the incoming gradient
                # to get the local gradients
                gradients += (config.backend.einsum(reordered_subscripts, gradient, *(_input for j, _input in enumerate(inputs) if j != i)),)

            return gradients


        if not all(isinstance(tensor, Tensor) for tensor in tensors):
            raise ValueError("Einsum is only supported when 'tensors' argument are Tensors")

        node = Operation(
            name='einsum',
            operation=lambda *tensors: config.backend.einsum(subscripts, *tensors),
            inputs=(self,) + tensors,
            grad_fn=_grad_fn
        )

        result = Tensor(
            requires_grad=(self.requires_grad or
                           any(tensor.requires_grad for tensor in tensors))
        )

        result.node = node

        return result


    def fire(self):
        if self.node is not None:
            # return computed Tensor
            # if operation is valid
            return self.node.fire()

        return self


    def zero_grad(
            self
    ):
        if self.requires_grad:
            self.grad = zeros_like(self.data).data

        else:
            self.grad = None


    def backward(
            self,
            gradient: Optional[ArrayType] = None
    ):
        if not self.requires_grad:
            return

        # grad: gradient of a loss with respect to the node's output 
        # is initialtized to 1: to start with computing
        # gradient of loss with respect to itself
        if gradient is None:
            gradient = ones_like(self.data).data

        # accumulate gradient at the current node: chain rule
        # inplace += is not used to avoid broadcasting error
        self.grad = self.grad + gradient # implicit broadcasting

        _gradient: ArrayType = copy.deepcopy(self.grad)

        if self.node is not None:
            self.node.backward(_gradient)


    def sprint(self):
        return "Tensor(data: {}, dtype: {}, grad: {}, requires_grad: {})".format(
            self.data, self.data_type.value_as_str, self.grad, self.requires_grad
        )


    def __repr__(self):
        return "Tensor(shape: {}, dtype: {}, requires_grad: {})".format(
            self.data.shape, self.data_type.value_as_str, self.requires_grad
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
            operation: Any,
            inputs: tuple,
            grad_fn: Any = None
    ):
        # value of node
        self.value = Tensor()

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
            self.inputs = tuple(node_input.fire() for node_input in self.inputs)

            # compute the values of the inputs to the operation
            # and generate a new Tensor
            self.value = Tensor(
                data=self.operation(*(tensor.data for tensor in self.inputs)),
                requires_grad=any(node_input.requires_grad for node_input in self.inputs)
            )

            self.fired = True

            # connect the new computed Tensor
            # to this Operation
            self.value.node = self

        return self.value


    def backward(
            self,
            gradient: ArrayType
    ):
        """
        Implement backpropogation: compute gradients for node
        in the computational graph
        """
        # compute gradients with respect to the node inputs
        # and propogate the gradients backward through the graph
        # by recursively calling backward() on the input nodes
        if self.grad_fn is not None:
            gradients: tuple = self.grad_fn(gradient,
                                            *(input_tensor.data for input_tensor in self.inputs))

            for node_input, input_grad in zip(self.inputs, gradients):
                node_input.backward(input_grad)


    def __repr__(self):
        return "Operation(op: {}, value_shape: {}, fired: {})".format(
            self.name, self.value.shape, self.fired
        )


# functions
@jaxtyped(typechecker=typechecker)
def zeros_like(
        tensor: Union[Tensor, ArrayType]
):
    if isinstance(tensor, ArrayType):
        return Tensor(
            data=config.backend.zeros_like(tensor),
            dtype=DataType(str(tensor.dtype))
        )

    else:
        return Tensor(
            data=config.backend.zeros_like(tensor.data),
            dtype=tensor.data_type,
            requires_grad=tensor.requires_grad
        )


@jaxtyped(typechecker=typechecker)
def ones_like(
        tensor: Union[Tensor, ArrayType]
):
    if isinstance(tensor, ArrayType):
        return Tensor(
            data=config.backend.ones_like(tensor),
            dtype=DataType(str(tensor.dtype))
        )

    else:
        return Tensor(
            data=config.backend.ones_like(tensor.data),
            dtype=tensor.data_type,
            requires_grad=tensor.requires_grad
        )


@jaxtyped(typechecker=typechecker)
def full(
        shape: tuple,
        fill_value: Union[int, float],
        dtype: Optional[DataType] = DataType('float32')
):
    return Tensor(
        data=config.backend.full(shape, fill_value),
        dtype=dtype
    )


# defined here to avoid circular import dependency
@jaxtyped(typechecker=typechecker)
def prod(
    array: Union[tuple, list]
): 
    return functools.reduce((lambda x, y: x * y), array)

