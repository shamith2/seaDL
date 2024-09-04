import torch
import torch.nn.functional as F
import numpy as np
import mlx.core as mx
import pytest

import seaML.nn as nn


@pytest.fixture
def pytest_configure():
    pytest.device = mx.gpu
    pytest.n_tests = 10


def test_nn_relu(pytest_configure: None):
    x = mx.random.normal(shape=(10,)) - 0.5
    x_torch = torch.from_numpy(np.array(x))

    relu = nn.ReLU()

    actual = relu(nn.Tensor(x)).fire()
    actual_torch = torch.from_numpy(np.array(actual))

    expected = F.relu(x_torch)

    torch.testing.assert_close(actual_torch, expected)

