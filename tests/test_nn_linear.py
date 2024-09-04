import torch
import numpy as np
import mlx.core as mx
import pytest

import seaML.nn as nn


@pytest.fixture
def pytest_configure():
    pytest.device = mx.gpu


def test_nn_flatten(pytest_configure):
    x = nn.Tensor(mx.arange(24, dtype=mx.float32)).reshape((2, 3, 4))

    assert nn.Flatten(start_dim=0)(x).fire().shape == (24,)
    assert nn.Flatten(start_dim=1)(x).fire().shape == (2, 12)
    assert nn.Flatten(start_dim=0, end_dim=1)(x).fire().shape == (6, 4)
    assert nn.Flatten(start_dim=0, end_dim=-2)(x).fire().shape == (6, 4)


def test_nn_linear(pytest_configure):
    x = mx.random.normal(shape=(10, 512))
    x_torch = torch.from_numpy(np.array(x))

    linear = nn.Linear(512, 64, bias=True, device=pytest.device)
    torch_linear = torch.nn.Linear(512, 64, bias=True)

    params = linear._parameters

    assert set(params.keys()) == {"weight", "bias"}

    assert linear.weight.shape == (64, 512)
    assert linear.bias.shape == (64,)

    linear.weight = nn.Tensor(np.array(torch_linear.weight.detach().clone()))
    linear.bias = nn.Tensor(np.array(torch_linear.bias.detach().clone()))

    actual = linear(nn.Tensor(x)).fire()
    actual_torch = torch.from_numpy(np.array(actual))

    expected = torch_linear(x_torch)

    torch.testing.assert_close(actual_torch, expected)


def test_nn_linear_no_bias(pytest_configure):
    x = mx.random.normal(shape=(10, 512))
    x_torch = torch.from_numpy(np.array(x))

    linear = nn.Linear(512, 64, bias=False, device=pytest.device)
    torch_linear = torch.nn.Linear(512, 64, bias=False)

    params = linear._parameters

    assert set(params.keys()) == {"weight"}

    assert linear.bias is None, "Bias should be None when not enabled"

    linear.weight = nn.Tensor(np.array(torch_linear.weight.detach().clone()))

    actual = linear(nn.Tensor(x)).fire()
    actual_torch = torch.from_numpy(np.array(actual))

    expected = torch_linear(x_torch)

    torch.testing.assert_close(actual_torch, expected)

