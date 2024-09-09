# __init__.py

__version__ = "0.2.0"

# config
from .config import config

# imports
from .base import Tensor, Operation, Device, DataType
from .base import zeros_like, ones_like
from .base import prod
