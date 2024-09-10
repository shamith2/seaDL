# config
import logging

logging.basicConfig(format='%(levelname)s:[config] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    def __init__(self):
        # default backend
        self.set_backend('mlx')

        # default device
        self.set_device('gpu')


    def set_backend(
            self,
            backend_library: str
    ):
        if backend_library not in ['mlx', 'numpy']:
            raise ValueError("Currently supported backends: mlx, numpy")

        # import backend based on backend_library
        if backend_library == 'mlx':
            import mlx.core as backend
            from mlx.core import array as Array
            from mlx.core import array as ArrayType
            import mlx.core as strided_lib

        elif backend_library == 'numpy':
            import numpy as backend
            from numpy import array as Array
            from numpy import ndarray as ArrayType
            import numpy.lib.stride_tricks as strided_lib


        # assign backend modules as config attributes
        self.backend_library = backend_library
        self.backend = backend
        self.strided_lib = strided_lib
        self.Array = Array
        self.ArrayType = ArrayType


    def set_device(
            self,
            device_type: str
    ):
        if self.backend_library == 'mlx':
            if device_type not in ['cpu', 'gpu']:
                raise ValueError("backend '{}' only supports: cpu, gpu".format(self.backend))

            device = self.backend.Device(self.get_device(device_type))
            self.backend.set_default_device(device)

            logging.info('using mlx with default device: {}\n'.format(device))

        elif self.backend_library == 'numpy':
            if device_type not in ['cpu']:
                raise ValueError("backend '{}' only supports: cpu".format(self.backend_library))

        self.device_type = device_type


    def get_device(
            self,
            device_type: str
    ):
        if self.backend_library == 'mlx':
            return self.backend.gpu if device_type == 'gpu' else self.backend.cpu

        else:
            return None


    def is_backend_mlx(self):
        return self.backend_library == 'mlx'


    def is_backend_numpy(self):
        return self.backend_library == 'numpy'


# global config instance
config = Config()

# global ArrayType
ArrayType = config.ArrayType

