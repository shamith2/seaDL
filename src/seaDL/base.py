from typing import Any, Union, Optional, Self
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker
import copy

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

    Incorporates lazy computation and backpropagation
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

        # node in the computational graph
        # that connects this Tensor
        self.node: Operation = None

        # whether to compute gradient for the node
        self.requires_grad = requires_grad

        # gradient with respect to this node
        if requires_grad and self.numel() != 0:
            self.grad = zeros_like(self.detach()).data

        else:
            self.grad = None


    def __array__(self):
        """
        For converting to numpy array,
        using numpy.array()
        """
        return NotImplementedError


    def numel(self) -> int:
        return self.data.size


    def __getitem__(
            self,
            indices
    ):
        """
        Support Tensor indexing

        Retrieve slices of elements from Tensor
        """
        def grad_fn(gradient: ArrayType):
            grad = zeros_like(gradient).data
            grad[indices] = gradient

            return (grad,)


        node = Operation(
            name='get_slice',
            operation=lambda x: x[indices],
            inputs=(self,),
            grad_fn=grad_fn
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
        if self.numel() == 0:
            fired_tensor = self.fire()

            # updated the Tensor after computation
            self.__dict__.update(fired_tensor.__dict__)

        _shape = self.data.shape

        self.strides = config.Array([1] * len(_shape))
        self.strides[:-1] = config.backend.cumprod(
                                config.Array(_shape[::-1])
                            )[::-1][1:]

        if config.is_backend_numpy():
            self.strides = self.strides * self.itemsize

        return self.strides.tolist()


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
        if isinstance(other, Tensor):
            node = Operation(
                name='mul',
                operation=lambda x, y: x * y,
                inputs=(self, other),
                grad_fn=lambda gradient, *inputs: (gradient * inputs[1],
                                                   gradient * inputs[0])
            )

            result = Tensor(
                requires_grad=(self.requires_grad or other.requires_grad)
            )

            # result of node
            result.node = node

        else:
            node = Operation(
                name='mul',
                operation=lambda x : x * other,
                inputs=(self,),
                grad_fn=lambda gradient, *inputs: (gradient,)
            )

            result = Tensor(
                dtype=self.data_type,
                requires_grad=self.requires_grad
            )

            # result of node
            result.node = node
        
        return result


    def __neg__(self):
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
        if isinstance(other, Tensor):
            node = Operation(
                name='__power__',
                operation=lambda x, y: config.backend.power(x, y),
                inputs=(self, other),
                grad_fn=lambda gradient, *inputs: (gradient * inputs[1] * config.backend.power(inputs[0], inputs[1] - 1),
                                                   gradient * config.backend.power(inputs[0], inputs[1]) * config.backend.log(inputs[0]))
            )

            result = Tensor(
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


    # overload right-hand atomic-like operations
    def __radd__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        return self.__add__(other)


    def __rsub__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        return -self.__sub__(other)


    # non-atomic operations
    def matmul(
            self,
            other: Self
    ):
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
            requires_grad=(self.requires_grad or other.requires_grad)
        )

        # result of node
        result.node = node
    
        return result


    def sqrt(self):
        return self.__pow__(0.5)


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
        def grad_fn(gradient, *tensors):
            grad = zeros_like(gradient).data
            grad[indices] = gradient

            return (grad,)

        def op(x, y):
            # x[indices] = y.astype(x.dtype)
            x[indices] = y

            return x


        node = Operation(
            name='set_slice',
            operation=op,
            inputs=(self, value),
            grad_fn=grad_fn
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
            axes: Optional[tuple] = None
    ):
        node = Operation(
            name='transpose',
            operation=lambda x: config.backend.transpose(x, axes),
            inputs=(self,),
            grad_fn=lambda gradient, *inputs: (gradient.T,)
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
        Return a new Tensor which is the reshape of self
        """
        node = Operation(
            name='reshape',
            operation=lambda x: config.backend.reshape(x, shape),
            inputs=(self,),
            grad_fn=lambda gradient, *inputs: (gradient,)
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
            grad_fn=lambda gradient, *inputs: (gradient,)
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
        if isinstance(other, Tensor):
            node = Operation(
                name='maximum',
                operation=lambda x, y: config.backend.maximum(x, y),
                inputs=(self, other),
                grad_fn=lambda gradient, *inputs: (gradient * config.backend.sign(inputs[0]),
                                                   gradient * config.backend.sign(inputs[0]))
            )

            result = Tensor(
                requires_grad=(self.requires_grad or other.requires_grad)
            )

            result.node = node

        else:
            node = Operation(
                name='maximum',
                operation=lambda x: config.backend.maximum(x, other),
                inputs=(self,),
                grad_fn=lambda gradient, *inputs: (gradient * config.backend.sign(inputs[0]),)
            )

            result = Tensor(
                requires_grad=self.requires_grad
            )

            result.node = node

        return result


    def einsum(
            self,
            subscripts: str,
            *tensors: Self
    ):
        if not all(isinstance(tensor, Tensor) for tensor in tensors):
            raise ValueError("Einsum is only supported when 'tensors' argument are Tensors")

        node = Operation(
            name='einsum',
            operation=lambda *tensors: config.backend.einsum(subscripts, *tensors),
            inputs=(self,) + tensors,
            grad_fn=None # Not yet Implemented
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

        if self.node is not None:
            # accumulate gradient at the current node: chain rule
            self.grad += gradient

            self.node.backward(gradient)


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


    def numerical_gradient_(
            self,
            h: float = 1e-6
    ):
        """
        Compute gradient of function fn at point p

        Inputs:
            fn = function to differentiate. Takes an array as input
            p: Array = Point to compute the gradient at
            h: float = Limit h -> 0

            gradient = (fn(p + h) - fn(p)) / h as h -> 0

        Outputs:
            gradient: Array = Gradient vector at p
        """
        for node_input in self.inputs:
            if isinstance(node_input, Tensor) and node_input.requires_grad:
                original_value = node_input

                print(original_value, node_input)

                node_input += h

                print(self.inputs)
                cc

                fn_plus = self.fire()

                node_input -= h

                fn_minus = self.fire()

                print(fn_plus, fn_minus)
                cc

                node_input = original_value

                node_input.grad = (fn_plus - fn_minus) / (2 * h)


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

