# __init__.py

__version__ = "0.2.2"

# config
from .config import config

# imports
from .engine import Tensor, Operation, Device, DataType
from .engine import zeros_like, ones_like
from .engine import prod, fire, no_grad
