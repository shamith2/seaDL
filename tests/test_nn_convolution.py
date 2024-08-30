import torch
import torch.nn.functional as F
import numpy as np
import mlx.core as mx
import pytest

from seaML.nn import Conv2d


@pytest.fixture
def pytest_configure():
    pytest.device = mx.gpu
    pytest.n_tests = 10
    pytest.tuples = True
    pytest.use_bias = True


def test_nn_conv2d(pytest_configure):
    '''
    Your weight should be called 'weight' and have an appropriate number of elements
    '''
    m = Conv2d(4, 5, 3)
    assert m.weight.size == 4 * 5 * 3 * 3

    for _ in range(pytest.n_tests):
        b = mx.random.randint(1, 10)
        h = mx.random.randint(10, 300)
        w = mx.random.randint(10, 300)
        ci = mx.random.randint(1, 20)
        co = mx.random.randint(1, 20)

        if pytest.tuples:
            stride = tuple(mx.random.randint(1, 5, shape=(2,)).tolist())
            padding = tuple(mx.random.randint(0, 5, shape=(2,)).tolist())
            kernel_size = tuple(mx.random.randint(1, 10, shape=(2,)).tolist())

        else:
            stride = mx.random.randint(1, 5).item()
            padding = mx.random.randint(0, 5).item()
            kernel_size = mx.random.randint(1, 10).item()

        x = mx.random.normal(shape=(b.item(), ci.item(), h.item(), w.item()))
        x_torch = torch.from_numpy(np.array(x))

        my_conv = Conv2d(
            in_channels=ci.item(),
            out_channels=co.item(),
            kernel_size=kernel_size,
            stride=stride,
            bias=pytest.use_bias,
            padding=padding,
            device=pytest.device
        )

        my_output = my_conv(x)
        my_output_torch = torch.from_numpy(np.array(my_output))

        torch_output = F.conv2d(
            x_torch,
            torch.from_numpy(np.array(my_conv.weight)),
            bias=torch.from_numpy(np.array(my_conv.bias)) if pytest.use_bias else None,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1
        )

        torch.testing.assert_close(my_output_torch, torch_output)

