import torch
import torch.nn.functional as F
import numpy as np
import mlx.core as mx
import pytest

from seaML.nn import ReLU


@pytest.fixture
def pytest_configure():
    pytest.device = mx.gpu
    pytest.n_tests = 10


def test_nn_relu(pytest_configure: None):
    x = mx.random.normal(shape=(10,)) - 0.5
    x_torch = torch.from_numpy(np.array(x))

    relu = ReLU(inplace=False, device=pytest.device)
    relu_inplace = ReLU(inplace=True)

    actual = relu(x)
    actual_torch = torch.from_numpy(np.array(actual))

    actual_inplace = relu_inplace(x)
    actual_inplace_torch = torch.from_numpy(np.array(actual_inplace))

    assert x is actual_inplace

    expected = F.relu(x_torch)

    torch.testing.assert_close(actual_torch, expected)
    torch.testing.assert_close(actual_inplace_torch, expected)

