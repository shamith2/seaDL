import torch
import numpy as np
import mlx.core as mx
import pytest

import seaDL
import seaDL.nn as nn


@pytest.fixture
def pytest_configure():
    pytest.device = None


def test_nn_flatten(pytest_configure):
    x = seaDL.Tensor(mx.arange(24)).reshape((2, 3, 4))

    actual_1 = nn.Flatten(start_dim=0)(x)
    actual_2 = nn.Flatten(start_dim=0, end_dim=-2)(x)

    seaDL.fire(actual_1)
    seaDL.fire(actual_2)

    assert actual_1.shape == (24,)
    assert actual_2.shape == (6, 4)


def test_nn_linear(pytest_configure):
    x = seaDL.random.normal(size=(2, 10, 512))
    x_torch = torch.from_numpy(np.array(x.data))

    linear = nn.Linear(512, 64, bias=True)
    torch_linear = torch.nn.Linear(512, 64, bias=True)

    einsum_linear = nn.Linear(512, 64, bias=True)

    params = linear._parameters

    assert set(params.keys()) == {"weight", "bias"}

    assert linear.weight.shape == (64, 512)
    assert linear.bias.shape == (64,)

    linear.weight = nn.Parameter(seaDL.Tensor(np.array(torch_linear.weight.detach().clone())))
    linear.bias = nn.Parameter(seaDL.Tensor(np.array(torch_linear.bias.detach().clone())))

    einsum_linear.weight = nn.Parameter(seaDL.Tensor(np.array(torch_linear.weight.detach().clone())))
    einsum_linear.bias = nn.Parameter(seaDL.Tensor(np.array(torch_linear.bias.detach().clone())))

    actual = linear(x, subscripts="bci,oi->bco")

    seaDL.fire(actual)

    actual_torch = torch.from_numpy(np.array(actual.data))

    actual_einsum = einsum_linear(x)

    seaDL.fire(actual_einsum)

    actual_einsum_torch = torch.from_numpy(np.array(actual_einsum.data))

    expected = torch_linear(x_torch)

    torch.testing.assert_close(actual_torch, expected)
    torch.testing.assert_close(actual_einsum_torch, expected)


def test_nn_linear_no_bias(pytest_configure):
    x = seaDL.random.normal(size=(10, 512))
    x_torch = torch.from_numpy(np.array(x.data))

    linear = nn.Linear(512, 64, bias=False)
    torch_linear = torch.nn.Linear(512, 64, bias=False)

    params = linear._parameters

    assert set(params.keys()) == {"weight"}

    assert linear.bias is None, "Bias should be None when not enabled"

    linear.weight = nn.Parameter(seaDL.Tensor(np.array(torch_linear.weight.detach().clone())))

    actual = linear(x)

    seaDL.fire(actual)

    actual_torch = torch.from_numpy(np.array(actual.data))

    expected = torch_linear(x_torch)

    torch.testing.assert_close(actual_torch, expected)

