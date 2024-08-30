import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mlx.core as mx
import pytest

from seaML.modules import pad1d_strided, pad2d_strided, conv1d_strided, conv2d_strided, maxpool2d_strided


@pytest.fixture
def pytest_configure():
    pytest.device = mx.gpu
    pytest.n_tests = 10


def test_pad1d(pytest_configure):
    '''
    Should work with one channel of width 4
    '''
    x = mx.arange(4, dtype=mx.float32).reshape((1, 1, 4))

    actual = pad1d_strided(x, 1, 3, -2.0, pytest.device)
    expected = mx.array([[[-2.0, 0.0, 1.0, 2.0, 3.0, -2.0, -2.0, -2.0]]])
    assert mx.allclose(actual, expected).all()

    actual = pad1d_strided(x, 1, 0, -2.0, pytest.device)
    expected = mx.array([[[-2.0, 0.0, 1.0, 2.0, 3.0]]])
    assert mx.allclose(actual, expected).all()


def test_pad1d_multi_channel(pytest_configure):
    '''
    Should work with two channels of width 2
    '''
    x = mx.arange(4, dtype=mx.float32).reshape((1, 2, 2))
    expected = mx.array([[[0.0, 1.0, -3.0, -3.0], [2.0, 3.0, -3.0, -3.0]]])

    actual = pad1d_strided(x, 0, 2, -3.0, pytest.device)
    assert mx.allclose(actual, expected).all()


def test_pad2d(pytest_configure):
    '''
    Should work with one channel of 2x2
    '''
    x = mx.arange(4, dtype=mx.float32).reshape((1, 1, 2, 2))
    expected = mx.array([[[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 3.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]]])


    actual = pad2d_strided(x, 2, 3, 0, 1, 0.0, pytest.device)
    assert mx.allclose(actual, expected).all()


def test_pad2d_multi_channel(pytest_configure):
    '''
    Should work with two channels of 2x1
    '''
    x = mx.arange(4, dtype=mx.float32).reshape((1, 2, 2, 1))
    expected = mx.array([[[[-1.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]], [[-1.0, 2.0], [-1.0, 3.0], [-1.0, -1.0]]]])
    
    actual = pad2d_strided(x, 0, 1, 1, 0, -1.0, pytest.device)
    assert mx.allclose(actual, expected).all()


def test_conv1d(pytest_configure):
    for _ in range(pytest.n_tests):
        b = mx.random.randint(1, 10)
        h = mx.random.randint(10, 300)
        ci = mx.random.randint(1, 20)
        co = mx.random.randint(1, 20)
        stride = mx.random.randint(1, 5)
        padding = mx.random.randint(0, 5)
        kernel_size = mx.random.randint(1, 10)

        x = mx.random.normal(shape=(b.item(), ci.item(), h.item()))
        x_torch = torch.from_numpy(np.array(x))

        weights = mx.random.normal(shape=(co.item(), ci.item(), kernel_size.item()))
        weights_torch = torch.from_numpy(np.array(weights))

        my_output = conv1d_strided(
            x,
            weights,
            stride=stride.item(),
            padding=padding.item(),
            device=pytest.device
        )

        my_output_torch = torch.from_numpy(np.array(my_output))

        torch_output = F.conv1d(x_torch, weights_torch, stride=stride.item(), padding=padding.item())

        torch.testing.assert_close(my_output_torch, torch_output, atol=1e-4, rtol=1e-5)


def test_conv2d(pytest_configure):
    for _ in range(pytest.n_tests):
        b = mx.random.randint(1, 10)
        h = mx.random.randint(10, 300)
        w = mx.random.randint(10, 300)
        ci = mx.random.randint(1, 20)
        co = mx.random.randint(1, 20)
        stride = tuple(mx.random.randint(1, 5, shape=(2,)).tolist())
        padding = tuple(mx.random.randint(0, 5, shape=(2,)).tolist())
        kernel_size = tuple(mx.random.randint(1, 10, shape=(2,)).tolist())
        
        x = mx.random.normal(shape=(b.item(), ci.item(), h.item(), w.item()), dtype=mx.float32)
        x_torch = torch.from_numpy(np.array(x))

        weights = mx.random.normal(shape=(co.item(), ci.item(), *kernel_size), dtype=mx.float32)
        weights_torch = torch.from_numpy(np.array(weights))

        my_output = conv2d_strided(
            x,
            weights,
            stride=stride,
            padding=padding,
            device=pytest.device
        )

        my_output_torch = torch.from_numpy(np.array(my_output))

        torch_output = torch.conv2d(x_torch, weights_torch, stride=stride, padding=padding)

        torch.testing.assert_close(my_output_torch, torch_output, atol=1e-4, rtol=1e-5)


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

        x = mx.random.normal(shape=(b.item(), ci.item(), h.item(), w.item()))
        x_torch = torch.from_numpy(np.array(x))

        my_output = maxpool2d_strided(
            x,
            kernel_size,
            stride=stride,
            padding=padding,
            device=pytest.device
        )

        my_output_torch = torch.from_numpy(np.array(my_output))

        torch_output = torch.max_pool2d(
            x_torch,
            kernel_size,
            stride=stride,
            padding=padding
        )

        torch.testing.assert_close(my_output_torch, torch_output)

