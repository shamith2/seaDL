# __init__.py

__version__ = "0.2.2"

# config
from .config import config, ArrayType

# imports
from .base import Device, DataType, GradientConfig, no_grad, prod
from .engine import Tensor, Operation, fire
from .engine import zeros_like, ones_like, full
