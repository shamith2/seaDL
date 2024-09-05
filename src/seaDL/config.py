# config

class Config:
    def __init__(self):
        # default backend
        self.set_backend('mlx')

        # default device
        self.set_device('cpu')


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

        elif self.backend_library == 'numpy':
            if device_type not in ['cpu']:
                raise ValueError("backend '{}' only supports: cpu".format(self.backend_library))

        self.device_type = device_type


    def get_device(
            self,
            device_type: str
    ):
        self.set_device(device_type)

        if self.backend == 'mlx':
            if self.device_type == 'cpu':
                return self.backend.cpu

            else:
                return self.backend.gpu

        else:
            return None


# global config instance
config = Config()

# global ArrayType
ArrayType = config.ArrayType

