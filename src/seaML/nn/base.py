from typing import Any, Union, Optional, Self
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from collections import OrderedDict
import numpy as np
import mlx.core as mx


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
            gradient: Optional[mx.array] = None
    ):
        """
        Implement backpropogation: compute gradients for node
        in the computational graph
        """
        # grad: gradient of a loss with respect to the node's output 
        # is initialtized to 1: to start with computing
        # gradient of loss with respect to itself
        if gradient is None:
            gradient = mx.ones_like(self.value)

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
class Tensor(mx.array):
    """
    Class that represents a Tensor datatype

    Each node is a Tensor in the computational graph

    Inherits from mlx.core.array class

    Incorporates lazy computation and backpropagation
    """
    def __init__(
            self,
            data: Union[int, float, list, tuple, np.ndarray, mx.array] = (),
            requires_grad: bool = True
    ):
        # initialize mlx.core.array with data
        super().__init__(data)

        # value of node: can be result of an operation or constant value
        if not isinstance(data, mx.array):
            self.data = mx.array(data)

        else:
            self.data = data

        # node in the computational graph
        # that connects this Tensor
        self.node: Operation = None

        # whether to compute gradient for the node
        self.requires_grad = requires_grad

        # gradient with respect to this node
        if requires_grad and self.numel() != 0:
            self.grad = mx.zeros_like(self.data)

        else:
            self.grad = None


    def __array__(self):
        return self.data


    def numel(self) -> int:
        return self.data.size


    @property
    def shape(self) -> tuple[int]:
        if self.numel() == 0:
            # when .shape is called,
            # if Tensor has not been computed,
            # compute the Tensor
            fired_tensor = self.fire()

            # updated the Tensor after computation
            self.__dict__.update(fired_tensor.__dict__)

        return self.data.shape


    @property
    def dtype(self) -> mx.Dtype:
        return self.data.dtype


    # overload atomic-like operations
    def __add__(
            self,
            other: Union[int, float, mx.array, Self]
    ):
        if isinstance(other, Tensor):
            node = Operation(
                name='__add__',
                operation=lambda x, y: mx.add(x, y),
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
                operation=lambda x: mx.add(x, other),
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
            other: Union[int, float, mx.array, Self]
    ):
        if isinstance(other, Tensor):
            node = Operation(
                name='__sub__',
                operation=lambda x, y: mx.subtract(x, y),
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
                operation=lambda x: mx.subtract(x, other),
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
            other: Union[int, float, mx.array, Self]
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
            operation=lambda x: mx.exp(x),
            inputs=(self,),
            grad_fn=lambda gradient: gradient * mx.exp(self.data)
        )

        result = Tensor(
            requires_grad=self.requires_grad
        )

        result.node = node

        return result


    def __pow__(
            self,
            other: Union[int, float, Self, mx.array]
    ):
        if isinstance(other, Tensor):
            node = Operation(
                name='__power__',
                operation=lambda x, y: mx.power(x, y),
                inputs=(self, other),
                grad_fn=lambda gradient: (gradient * other * mx.power(self.data, other - Tensor(1)),
                                          gradient * mx.power(self.data, other.data) * mx.log(self.data))
            )

            result = Tensor(
                requires_grad=(self.requires_grad or other.requires_grad)
            )

            # result of node
            result.node = node

        else:
            node = Operation(
                name='__power__',
                operation=lambda x: mx.power(x, other),
                inputs=(self,),
                grad_fn=lambda gradient: gradient * other * mx.power(self.data, other - 1)
            )

            result = Tensor(
                requires_grad=self.requires_grad
            )

            result.node = node

        return result


    # overload right-hand atomic-like operations
    def __radd__(
            self,
            other: Union[int, float, mx.array, Self]
    ):
        return self.__add__(other)


    def __rsub__(
            self,
            other: Union[int, float, mx.array, Self]
    ):
        return self.__sub__(-other)


    def __rmul__(
            self,
            other: Union[int, float, mx.array, Self]
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
            operation=lambda x, y: mx.matmul(x, y),
            inputs=(self, other),
            grad_fn=lambda gradient: (mx.matmul(gradient, mx.transpose(other.data)),
                                      mx.matmul(mx.transpose(self.data), gradient))
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
            operation=lambda x: mx.transpose(x, axes),
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
            operation=lambda x: mx.reshape(x, shape),
            inputs=(self,),
            grad_fn=lambda gradient: gradient
        )

        result = Tensor(
            requires_grad=self.requires_grad
        )

        result.node = node

        return result

        # return Tensor(
        #     data=mx.reshape(self.data, shape),
        #     requires_grad=self.requires_grad
        # )


    def maximum(
            self,
            other: Union[int, float, Self]
    ):
        if isinstance(other, Tensor):
            node = Operation(
                name='maximum',
                operation=lambda x, y: mx.maximum(x, y),
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
                operation=lambda x: mx.maximum(x, other),
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
            operation=lambda ts: mx.einsum(subscripts,
                                           *(t for t in ts)),
            inputs=(self,) + tensors,
            grad_fn=None
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
            gradient: Union[mx.array, None] = None
    ):
        if not self.requires_grad:
            return

        # grad: gradient of a loss with respect to the node's output 
        # is initialtized to 1: to start with computing
        # gradient of loss with respect to itself
        if gradient is None:
            gradient = mx.ones_like(self.data)

        if self.node is not None:
            # accumulate gradient at the current node: chain rule
            self.grad = gradient

            self.node.backward(gradient)


    def __repr__(self):
        return "Tensor(shape: {}, requires_grad: {})".format(
            self.data.shape, self.requires_grad
        )


@jaxtyped(typechecker=typechecker)
class Parameter(Tensor):
    def __init__(
            self,
            data: Union[int, float, np.ndarray, mx.array],
            requires_grad: bool = True
    ):
        if not isinstance(data, mx.array):
            data = mx.array(data)

        super().__init__(data=data, requires_grad=requires_grad)


    def zero_grad(self):
        """
        Initialize gradient for Parameter to zero
        """
        if self.requires_grad:
            self.grad = mx.zeros_like(self.data)


    def _init_graph(
            self,
            inputs,
            operation
    ):
        """
        Initialize node in computational graph
        """
        self.inputs = inputs

        self.operation = operation

        # return the current Parameter instance
        return self


    def __repr__(self):
        return "Parameter(shape: {}, requires_grad: {})".format(
            self.data.shape, self.requires_grad
        )


@jaxtyped(typechecker=typechecker)
class Module:
    """
    Like torch.nn.Module
    """
    def __init__(self):
        # for storing submodules: layers in a neural network
        self._modules: OrderedDict = OrderedDict()

        # for storing parameters: weights and bias of layers in the network
        self._parameters: OrderedDict = OrderedDict()

        # for storing trainable parameters
        self._trainable_parameters: OrderedDict = OrderedDict()

        # for storing buffers
        self._buffers: OrderedDict = OrderedDict()

        # state dict
        self._state: OrderedDict = OrderedDict()

        # training mode
        self.training = True


    def __call__(
            self,
            *inputs
    ):
        """
        Method for allowing a module to be called as a fn()
        """
        raise NotImplementedError


    def __setattr__(
            self,
            name: str,
            value: Union[Self, Parameter]
    ):
        """
        Implictly add Parameter or Module to the dict of parameters or submodules
        without explictly calling self.add_modules or self.register_paramater
        """
        if isinstance(value, Parameter):
            self._parameters[name] = value

            if value.requires_grad:
                self._trainable_parameters[name] = value

        elif isinstance(value, Module):
            self._modules[name] = value

        super().__setattr__(name, value)


    def state_dict(self):
        """
        Dict containing state: parameters, ...
        for Module and its submodules
        """
        # update parameters
        self._state.update(self._parameters)

        # update buffers
        self._state.update(self._buffers)

        # update submodules state
        for module in self._modules.values():
            self._state.update(module._state)


    def add_module(
            self,
            name: str,
            module: Self
    ):
        """
        Add a submodule to the dict of modules
        """
        self._modules[name] = module


    def register_paramater(
            self,
            name: str,
            parameter: Parameter
    ):
        """
        Register parameters for Module
        """
        self._parameters[name] = parameter

        if parameter.requires_grad:
            self._trainable_parameters[name] = parameter

        setattr(self, name, parameter)


    def register_buffer(
            self,
            name: str,
            value: mx.array
    ):
        """
        Register buffers for Module
        """
        self._buffers[name] = value

        if name in self._parameters.keys():
            del self._parameters[name]

        setattr(self, name, value)


    def named_modules(self):
        """
        Generator: all submodules in Module

        Useful for listing all name and value of submodules in a Module
        """
        for name, module in self._modules.items():
            yield (name, module)


    def parameters(self):
        """
        Generator: all parameter of Module
        """
        # yield parameters from current module
        for parameter in self._parameters.values():
            yield parameter

        # recursively yield parameters from all submodules
        for module in self._modules.values():
            yield from module.paramters()


    def trainable_parameters(self):
        """
        Generator: all parameter of Module
        """
        # yield parameters from current module
        for name, parameter in self._trainable_parameters.values():
            yield (name, parameter)

        # recursively yield parameters from all submodules
        for module in self._modules.values():
            yield from module.trainable_parameters()


    def named_parameters(self):
        """
        Generator: all parameters of Module

        Useful for listing all name and value of parameters in a Module
        """
        # yield parameters from current module
        for name, parameter in self._parameters.items():
            yield (name, parameter)

        # recursively yield parameters from all submodules
        for module in self._modules.values():
            yield from module.named_parameters()


    def named_buffers(self):
        """
        Generator: all buffers of Module
        """
        for name, buffer in self._buffers.items():
            yield name, buffer

        # recursively yield parameters from all submodules
        for module in self._modules.values():
            yield from module.named_buffers()


    def children(self):
        """
        Return iterator over all the submodules in the module

        Useful for iterating over all submodules when performing operations
        on submodules like initialization
        """
        return self._modules.values()


    def apply(
            self,
            fn
    ):
        """
        Apply a fn to a Module and all its submodules recursively

        Useful for performing operations: fn to a Module and its submodules
        like weight initialization, freezing parameters
        """
        for module in self.children():
            module.apply(fn)

        fn(self)


    def train(self):
        """
        Set Module to training mode
        """
        self.apply(lambda submodule: setattr(submodule, 'training', True))


    def eval(self):
        """
        Set Module to inference mode
        """
        self.apply(lambda submodule: setattr(submodule, 'training', False))


    def requires_grad_(
            self,
            requires_grad: bool
    ):
        """
        Recursively set requires_grad for all parameters in the Module
        """
        for param in self._parameters.values():
            param.requires_grad = requires_grad

        for child in self.children():
            child.requires_grad_(requires_grad)

    def extra_repr(self):
        raise NotImplementedError

