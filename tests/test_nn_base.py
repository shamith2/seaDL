import pytest
import numpy as np

from seaDL import Tensor
import seaDL.nn as nn
import seaDL.random as random


@pytest.fixture
def pytest_configure():
    pytest.device = None
    pytest.n_tests = 10


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(3, 2)
        self.relu = nn.ReLU()
    
    def __call__(self, x):
        out = self.linear(x)
        return self.relu(out)


def test_parameter(pytest_configure):
    param1 = nn.Parameter(Tensor([1.0, -2.0, 3.0]))
    param2 = nn.Parameter(Tensor([4.0, 5.0, 6.0]))

    result_neg_1 = -param1

    result_add_1 = param1 + param2
    result_add_2 = param1 + 2.0
    result_add_3 = 3.0 + param2

    result_sub_1 = param1 - param2
    result_sub_2 = param1 - 3.0
    result_sub_3 = 3.0 - param2

    result_mul_1 = param1 * param2
    result_mul_2 = param1 * 5.0
    result_mul_3 = -4.0 * param2

    result_pow_1 = param1 ** 2
    result_pow_2 = param1 ** param2

    np.testing.assert_allclose(np.array(result_neg_1.fire().data), np.array([-1.0, 2.0, -3.0]))   

    np.testing.assert_allclose(np.array(result_add_1.fire().data), np.array([5.0, 3.0, 9.0]))
    np.testing.assert_allclose(np.array(result_add_2.fire().data), np.array([3.0, 0.0, 5.0]))
    np.testing.assert_allclose(np.array(result_add_3.fire().data), np.array([7.0, 8.0, 9.0]))

    np.testing.assert_allclose(np.array(result_sub_1.fire().data), np.array([-3.0, -7.0, -3.0]))
    np.testing.assert_allclose(np.array(result_sub_2.fire().data), np.array([-2.0, -5.0, 0.0]))
    np.testing.assert_allclose(np.array(result_sub_3.fire().data), np.array([-1.0, -2.0, -3.0]))

    np.testing.assert_allclose(np.array(result_mul_1.fire().data), np.array([4.0, -10.0, 18.0]))
    np.testing.assert_allclose(np.array(result_mul_2.fire().data), np.array([5.0, -10.0, 15.0]))
    np.testing.assert_allclose(np.array(result_mul_3.fire().data), np.array([-16.0, -20.0, -24.0]))

    np.testing.assert_allclose(np.array(result_pow_1.fire().data), np.array([1.0, 4.0, 9.0]))
    np.testing.assert_allclose(np.array(result_pow_2.fire().data), np.array([1.0, -32.0, 729.0]))


def test_module(pytest_configure):
    net = SimpleNet()

    x = random.normal(shape=(3,))

    output = net(x)

    c_output = output.fire()

    np.testing.assert_allclose(np.array(c_output.data),
                               np.maximum(np.array(((x.data @ net.linear.weight.data.transpose()) + net.linear.bias.data)), 0.0))

