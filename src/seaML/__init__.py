# __init__.py

__version__ = "0.1.0"

# config
from .config import config

# imports
from .base import Tensor, Operation, Device, DataType
from .base import zeros_like, ones_like
