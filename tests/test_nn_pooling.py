import torch
import torch.nn as nn
import numpy as np
import mlx.core as mx
import pytest

import seaDL
from seaDL.nn import MaxPool2d, AveragePool2d


@pytest.fixture
def pytest_configure():
    pytest.device = None
    pytest.n_tests = 10
    pytest.tuples = True
    pytest.use_bias = True


def test_nn_maxpool2d(pytest_configure):
    for _ in range(pytest.n_tests):
        b = mx.random.randint(1, 10)
        h = mx.random.randint(10, 50)
        w = mx.random.randint(10, 50)
        ci = mx.random.randint(1, 20)

        if pytest.tuples:
            stride = tuple(mx.random.randint(1, 5, shape=(2,)).tolist())
            kernel_size = tuple(mx.random.randint(1, 10, shape=(2,)).tolist())
            kH, kW = kernel_size
            padding = (mx.random.randint(0, 1 + kH // 2).item(), mx.random.randint(0, 1 + kW // 2).item())

        else:
            stride = mx.random.randint(1, 5)
            kernel_size = mx.random.randint(1, 10)
            padding = mx.random.randint(0, 1 + kernel_size // 2)

        x = seaDL.random.normal(size=(b.item(), ci.item(), h.item(), w.item()))
        x_torch = torch.from_numpy(np.array(x.data))

        my_maxpool2d = MaxPool2d(
            kernel_size,
            stride=stride,
            padding=padding
        )

        my_output = my_maxpool2d(x).fire()
        my_output_torch = torch.from_numpy(np.array(my_output.data))

        torch_output = nn.MaxPool2d(
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            ceil_mode=False
        )(x_torch)

        torch.testing.assert_close(my_output_torch, torch_output)


def test_averagepool(pytest_configure):
    x = seaDL.Tensor(mx.arange(24)).reshape((1, 2, 3, 4))

    actual = AveragePool2d()(x).fire()

    expected = np.array([[5.5, 17.5]])
    np.testing.assert_allclose(np.array(actual.data), expected)

