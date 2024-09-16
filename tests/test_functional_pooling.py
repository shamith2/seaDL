import torch
import torch.nn.functional as F
import numpy as np
import mlx.core as mx
import pytest

import seaDL
import seaDL.nn as nn


@pytest.fixture
def pytest_configure():
    pytest.device = None
    pytest.n_tests = 10


def test_maxpool2d(pytest_configure):
    for _ in range(pytest.n_tests):
        b = mx.random.randint(1, 10)
        h = mx.random.randint(10, 50)
        w = mx.random.randint(10, 50)
        ci = mx.random.randint(1, 20)

        stride = tuple(mx.random.randint(1, 5, shape=(2,)).tolist())
        kernel_size = tuple(mx.random.randint(1, 10, shape=(2,)).tolist())
        kH, kW = kernel_size
        padding = (mx.random.randint(0, 1 + kH // 2).item(), mx.random.randint(0, 1 + kW // 2).item())

        x = seaDL.random.normal(size=(b.item(), ci.item(), h.item(), w.item()))
        x_torch = torch.from_numpy(np.array(x.data))

        my_output = nn.functional.max_pool2d(
            x,
            kernel_size,
            stride=stride,
            padding=padding
        )

        seaDL.fire(my_output)

        my_output_torch = torch.from_numpy(np.array(my_output.data))

        torch_output = F.max_pool2d(
            x_torch,
            kernel_size,
            stride=stride,
            padding=padding
        )

        torch.testing.assert_close(my_output_torch, torch_output)

