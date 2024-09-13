import pytest
import torch
import torch.nn.functional as F
import numpy as np

import seaDL
import seaDL.nn as nn


@pytest.fixture
def pytest_configure():
    pytest.device = None
    pytest.n_tests = 10


def test_nn_relu(pytest_configure):
    x = nn.Parameter(seaDL.random.normal(size=(10,)) - 0.5)
    x_torch = torch.from_numpy(np.array(x.data))

    relu = nn.ReLU()

    actual = relu(x)
    seaDL.fire(actual)

    actual_torch = torch.from_numpy(np.array(actual.data))

    expected = F.relu(x_torch)

    torch.testing.assert_close(actual_torch, expected)


def test_nn_softmax(pytest_configure):
    x = nn.Parameter(seaDL.random.normal(size=(2, 4)) - 0.5)
    x_torch = torch.from_numpy(np.array(x.data))

    softmax = nn.Softmax(dim=0)

    actual = softmax(x)
    seaDL.fire(actual)

    actual_torch = torch.from_numpy(np.array(actual.data))

    expected = F.softmax(x_torch, dim=0)

    torch.testing.assert_close(actual_torch, expected)

    softmax = nn.Softmax(dim=1)

    actual = softmax(x)
    seaDL.fire(actual)

    actual_torch = torch.from_numpy(np.array(actual.data))

    expected = F.softmax(x_torch, dim=1)

    torch.testing.assert_close(actual_torch, expected)


def test_nn_log_softmax(pytest_configure):
    x = nn.Parameter(seaDL.random.normal(size=(2, 4)) - 0.5)
    x_torch = torch.from_numpy(np.array(x.data))

    softmax = nn.LogSoftmax(dim=0)

    actual = softmax(x)
    seaDL.fire(actual)

    actual_torch = torch.from_numpy(np.array(actual.data))

    expected = F.log_softmax(x_torch, dim=0)

    torch.testing.assert_close(actual_torch, expected)

    softmax = nn.LogSoftmax(dim=1)

    actual = softmax(x)
    seaDL.fire(actual)

    actual_torch = torch.from_numpy(np.array(actual.data))

    expected = F.log_softmax(x_torch, dim=1)

    torch.testing.assert_close(actual_torch, expected)

