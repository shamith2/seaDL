import pytest
import numpy as np

import seaDL
from seaDL import Tensor, config
import seaDL.nn as nn
import seaDL.random
from seaDL.utils import gradient_check


@pytest.fixture
def pytest_configure():
    pytest.device = None
    pytest.n_tests = 10


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(3, 2)

        self.linear.weight = nn.Parameter(Tensor([[0.1, 0.2, -0.9], [-1.4, 0.5, 0.7]]))
        self.linear.bias = nn.Parameter(Tensor([-1.0, 2.0]))

        self.relu = nn.ReLU()
    
    def __call__(self, x):
        out = self.linear(x, subscripts="bi,oi->bo")
        y = self.relu(out)

        return y


def test_parameter(pytest_configure):
    param1 = nn.Parameter(Tensor([1.0, -2.0, 3.0]))
    param2 = nn.Parameter(Tensor([4.0, 5.0, 6.0]))

    result_neg_1 = -param1

    result_add_1 = param1 + param2
    result_add_2 = param1 + 2.0

    result_sub_1 = param1 - param2
    result_sub_2 = param1 - 3.0

    result_mul_1 = param1.mul(param2)
    result_mul_2 = param2.mul(5.0)

    result_div_1 = param1.div(param2)
    result_div_2 = param2.div(5.0)

    result_pow_1 = param1 ** 2
    result_pow_2 = param1 ** param2

    np.testing.assert_allclose(np.array(result_neg_1.fire().data), np.array([-1.0, 2.0, -3.0]))   

    np.testing.assert_allclose(np.array(result_add_1.fire().data), np.array([5.0, 3.0, 9.0]))
    np.testing.assert_allclose(np.array(result_add_2.fire().data), np.array([3.0, 0.0, 5.0]))

    np.testing.assert_allclose(np.array(result_sub_1.fire().data), np.array([-3.0, -7.0, -3.0]))
    np.testing.assert_allclose(np.array(result_sub_2.fire().data), np.array([-2.0, -5.0, 0.0]))

    np.testing.assert_allclose(np.array(result_mul_1.fire().data), np.array([4.0, -10.0, 18.0]))
    np.testing.assert_allclose(np.array(result_mul_2.fire().data), np.array([20.0, 25.0, 30.0]))

    np.testing.assert_allclose(np.array(result_div_1.fire().data), np.array([0.25, -0.4, 0.5]))
    np.testing.assert_allclose(np.array(result_div_2.fire().data), np.array([0.8, 1.0, 1.2]))

    np.testing.assert_allclose(np.array(result_pow_1.fire().data), np.array([1.0, 4.0, 9.0]))
    np.testing.assert_allclose(np.array(result_pow_2.fire().data), np.array([1.0, -32.0, 729.0]))


def test_module(pytest_configure):
    net = SimpleNet()

    x = Tensor([[0.5, 2.5, -3.4]])

    output = net(x)

    c_output = output.fire()

    c_output.backward()

    np.testing.assert_allclose(np.array(c_output.data),
                               np.maximum(np.array(((x.data @ net.linear.weight.data.transpose()) + net.linear.bias.data)), 0),
                               rtol=1e-5)

    assert gradient_check(c_output, net.linear.weight, h=1e-6, error_tolerance=0.03)
    assert gradient_check(c_output, net.linear.bias, h=1e-6, error_tolerance=0.03)


def test_auto_diff_1(pytest_configure):
    x = Tensor([[[[2.0, 3.0, 4.0, 5.0], [-2.0, -3.0, -4.0, -5.0]]], [[[2.0, 3.0, 4.0, 5.0], [-2.0, -3.0, -4.0, -5.0]]]])

    w = Tensor([[[[2.0, 3.0, 4.0, 5.0], [-2.0, -3.0, -4.0, -5.0]]], [[[2.0, 3.0, 4.0, 5.0], [-2.0, -3.0, -4.0, -5.0]]]], requires_grad=True)

    einsum_op = w.einsum("ijmn->j").fire()

    einsum_op.backward()

    assert gradient_check(einsum_op, w, h=1e-6, error_tolerance=0.03)

    w.zero_grad()

    einsum_op = x.einsum("ijkm,ijkn->in", w).fire()

    einsum_op.backward()

    assert gradient_check(einsum_op, w, h=1e-6, error_tolerance=0.06)


def test_auto_diff_2(pytest_configure):
    w = Tensor([[[[2.0, 3.0, 4.0, 5.0], [-2.0, -3.0, -4.0, -5.0]]], [[[2.0, 3.0, 4.0, 5.0], [-2.0, -3.0, -4.0, -5.0]]]], requires_grad=True)

    sq_op = w.squeeze(dim=(1,)).fire()

    sq_op.backward()

    assert gradient_check(sq_op, w, h=1e-6, error_tolerance=0.03)

    w.zero_grad()

    unsq_op = w.unsqueeze(dim=(0,)).fire()

    unsq_op.backward()

    assert gradient_check(unsq_op, w, h=1e-6, error_tolerance=0.03)

    w.zero_grad()

    mean_op = w.mean(dim=(1, 2)).fire()

    mean_op.backward()

    assert gradient_check(mean_op, w, h=1e-6, error_tolerance=0.03)

    w.zero_grad()

    max_op = w.max(dim=(-3, -1)).fire()

    max_op.backward()

    assert gradient_check(max_op, w, h=1e-6, error_tolerance=0.03)

