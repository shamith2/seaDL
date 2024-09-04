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
        if dtype.lower() == 'float32':
            self.value = config.backend.float32

        elif dtype.lower() == 'float16':
            self.value = config.backend.float16

        else:
            raise ValueError("dtype is not a valid datatype or is not supported")


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

        self.requires_grad = any(node_input.requires_grad for node_input in inputs)

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
                requires_grad=self.requires_grad
            )

            self.fired = True

            # connect the new computed Tensor
            # to this Operation
            self.value.node = self

        return self.value


    def backward(
            self,
            gradient: Optional[ArrayType] = None
    ):
        """
        Implement backpropogation: compute gradients for node
        in the computational graph
        """
        # grad: gradient of a loss with respect to the node's output 
        # is initialtized to 1: to start with computing
        # gradient of loss with respect to itself
        if gradient is not None:
            gradient = ones_like(self.detach())

        # compute gradients with respect to the node inputs
        # and propogate the gradients backward through the graph
        # by recursively calling backward() on the input nodes
        if self.grad_fn is not None:
            gradients: tuple = self.grad_fn(gradient)

            for node_input, input_grad in zip(self.inputs, gradients):
                node_input.backward(input_grad)


    def gradient_fn(
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
            requires_grad: bool = False
    ):
        # value of node: can be result of an operation or constant value
        if not isinstance(data, ArrayType):
            self.data = config.Array(data)

        else:
            self.data = data

        # node in the computational graph
        # that connects this Tensor
        self.node: Operation = None

        # whether to compute gradient for the node
        self.requires_grad = requires_grad

        # gradient with respect to this node
        if requires_grad and self.numel() != 0:
            self.grad = zeros_like(self.detach())

        else:
            self.grad = None


    def __array__(self):
        """
        For converting to numpy array,
        using numpy.array()
        """
        return self.data


    def numel(self) -> int:
        return self.data.size


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
        return self.data.dtype


    def astype(
            self,
            dtype: DataType
    ):
        return Tensor(
            data=self.data.astype(dtype.value),
            requires_grad=self.requires_grad
        )


    def detach(self):
        return Tensor(
            data=copy.deepcopy(self.data),
            requires_grad=False
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
                grad_fn=lambda gradient: (gradient, gradient)
            )

            result = Tensor(
                requires_grad=(self.requires_grad or other.requires_grad)
            )

            # result of node
            result.node = node

        else:
            node = Operation(
                name='__add__',
                operation=lambda x: config.backend.add(x, other),
                inputs=(self,),
                grad_fn=lambda gradient: gradient
            )

            result = Tensor(
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
                grad_fn=lambda gradient: (gradient, -gradient)
            )

            result = Tensor(
                requires_grad=(self.requires_grad or other.requires_grad)
            )

            # result of node
            result.node = node

        else:
            node = Operation(
                name='__sub__',
                operation=lambda x: config.backend.subtract(x, other),
                inputs=(self,),
                grad_fn=lambda gradient: gradient
            )

            result = Tensor(
                requires_grad=self.requires_grad
            )

            result.node = node

        return result


    def __mul__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        if isinstance(other, Tensor):
            node = Operation(
                name='__mul__',
                operation=lambda x, y: x * y,
                inputs=(self, other),
                grad_fn=lambda gradient: (gradient * other.data,
                                          gradient, self.data)
            )

            result = Tensor(
                requires_grad=(self.requires_grad or other.requires_grad)
            )

            # result of node
            result.node = node

        else:
            node = Operation(
                name='__mul__',
                operation=lambda x : x * other,
                inputs=(self,),
                grad_fn=lambda gradient: gradient
            )

            result = Tensor(
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
            grad_fn=lambda gradient: gradient
        )

        result = Tensor(
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
            grad_fn=lambda gradient: gradient * config.backend.exp(self.data)
        )

        result = Tensor(
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
                grad_fn=lambda gradient: (gradient * other * config.backend.power(self.data, other - Tensor(1)),
                                          gradient * config.backend.power(self.data, other.data) * config.backend.log(self.data))
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
                grad_fn=lambda gradient: gradient * other * config.backend.power(self.data, other - 1)
            )

            result = Tensor(
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
        return self.__sub__(-other)


    def __rmul__(
            self,
            other: Union[int, float, ArrayType, Self]
    ):
        return self.__mul__(other)


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
            grad_fn=lambda gradient: (config.backend.matmul(gradient, config.backend.transpose(other.data)),
                                      config.backend.matmul(config.backend.transpose(self.data), gradient))
        )

        result = Tensor(
            requires_grad=(self.requires_grad or other.requires_grad)
        )

        # result of node
        result.node = node
    
        return result


    # transformation operations
    def transpose(
            self,
            axes: Optional[tuple] = None
    ):
        node = Operation(
            name='transpose',
            operation=lambda x: config.backend.transpose(x, axes),
            inputs=(self,),
            grad_fn=lambda gradient: gradient
        )

        result = Tensor(
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
            grad_fn=lambda gradient: gradient
        )

        result = Tensor(
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
                grad_fn=lambda gradient: (gradient, gradient)
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
                grad_fn=lambda gradient: gradient
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
            gradient: Union[ArrayType, None] = None
    ):
        if not self.requires_grad:
            return

        # grad: gradient of a loss with respect to the node's output 
        # is initialtized to 1: to start with computing
        # gradient of loss with respect to itself
        if gradient is None:
            gradient = ones_like(self.detach())

        if self.node is not None:
            # accumulate gradient at the current node: chain rule
            self.grad = gradient

            self.node.backward(gradient)


    def __repr__(self):
        return "Tensor(shape: {}, requires_grad: {})".format(
            self.data.shape, self.requires_grad
        )


# functions
@jaxtyped(typechecker=typechecker)
def zeros_like(
        tensor: Tensor
):
    return Tensor(
        data=config.backend.zeros_like(tensor.data),
        requires_grad=tensor.requires_grad
    )


@jaxtyped(typechecker=typechecker)
def ones_like(
        tensor: Tensor
):
    return Tensor(
        data=config.backend.ones_like(tensor.data),
        requires_grad=tensor.requires_grad
    )

