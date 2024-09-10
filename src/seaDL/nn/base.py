from typing import Any, Union, Self
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from collections import OrderedDict

from ..config import config, ArrayType
from ..base import Tensor, zeros_like


@jaxtyped(typechecker=typechecker)
class Parameter(Tensor):
    def __init__(
            self,
            data: Any,
            requires_grad: bool = True
    ):
        if not isinstance(data, Tensor):
            raise ValueError("Parameter only supports Tensor data")

        super().__init__(
            data=data.data,
            dtype=data.data_type,
            requires_grad=requires_grad
        )


    def zero_grad(self):
        """
        Initialize gradient for Parameter to zero
        """
        if self.requires_grad:
            self.grad = zeros_like(self.detach()).data


    def __repr__(self):
        return "Parameter(shape: {}, dtype: {}, requires_grad: {})".format(
            self.data.shape, self.data_type.value_as_str, self.requires_grad
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
            value: Union[Self, Tensor]
    ):
        """
        Implictly add Tensor or Module to the dict of parameters or submodules
        without explictly calling self.add_modules or self.register_paramater
        """
        if isinstance(value, Tensor):
            self._parameters[name] = value

            value.requires_grad = True

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
            value: Tensor
    ):
        """
        Register parameters for Module
        """
        self._parameters[name] = value

        value.requires_grad = True

        self._trainable_parameters[name] = value

        setattr(self, name, value)


    def register_buffer(
            self,
            name: str,
            value: Tensor
    ):
        """
        Register buffers for Module
        """
        self._buffers[name] = value

        if value.requires_grad:
            value.requires_grad = False

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


    def zero_grad(self):
        """
        Recursively initialize gradient for all parameters in the Module
        to zero
        """
        for param in self._parameters.values():
            param.zero_grad()

        for child in self.children():
            child.zero_grad()


    def extra_repr(self):
        raise NotImplementedError

