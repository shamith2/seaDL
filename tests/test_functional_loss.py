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


def test_loss_nll(pytest_configure):
    x = nn.Parameter(seaDL.random.normal(size=(4, 4)) - 0.5)
    x_torch = torch.from_numpy(np.array(x.data))

    softmax = nn.LogSoftmax(dim=-1)

    actual = softmax(x)

    labels = seaDL.Tensor([2, 1, 3, 0], dtype=seaDL.DataType('int32'))
    labels_torch = torch.tensor([2, 1, 3, 0])

    actual = nn.functional.nll_loss(actual, labels, reduction='mean')

    if seaDL.config.is_backend_numpy():
        actual = actual.squeeze(dim=0)

    seaDL.fire(actual)

    actual_torch = torch.from_numpy(np.array(actual.data))

    expected = F.nll_loss(F.log_softmax(x_torch, dim=-1), labels_torch, reduction='mean')

    torch.testing.assert_close(actual_torch, expected)


def test_loss_cross_entropy(pytest_configure):
    x = nn.Parameter(seaDL.random.normal(size=(4, 4)) - 0.5)
    x_torch = torch.from_numpy(np.array(x.data))

    labels = seaDL.Tensor([2, 1, 3, 0])
    labels_torch = torch.tensor([2, 1, 3, 0])

    actual = nn.functional.cross_entropy(x, labels, reduction='mean')

    if seaDL.config.is_backend_numpy():
        actual = actual.squeeze(dim=0)

    seaDL.fire(actual)

    actual_torch = torch.from_numpy(np.array(actual.data))

    expected = F.cross_entropy(x_torch, labels_torch, reduction='mean')

    torch.testing.assert_close(actual_torch, expected)

