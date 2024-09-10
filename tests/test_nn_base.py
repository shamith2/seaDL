import pytest
import numpy as np

import seaDL
import seaDL.nn as nn
from seaDL.utils import gradient_check


@pytest.fixture
def pytest_configure():
    pytest.device = None
    pytest.n_tests = 10


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(3, 2)

        self.linear.weight = nn.Parameter(seaDL.Tensor([[0.1, 0.2, -0.9], [-1.4, 0.5, 0.7]]))
        self.linear.bias = nn.Parameter(seaDL.Tensor([-1.0, 2.0]))

        self.relu = nn.ReLU()
    
    def __call__(self, x):
        out = self.linear(x, subscripts="bi,oi->bo")
        y = self.relu(out)

        return y


def test_parameter(pytest_configure):
    param1 = nn.Parameter(seaDL.Tensor([1.0, -2.0, 3.0]))
    param2 = nn.Parameter(seaDL.Tensor([4.0, 5.0, 6.0]))

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

    seaDL.fire(result_neg_1)

    seaDL.fire(result_add_1)
    seaDL.fire(result_add_2)

    seaDL.fire(result_sub_1)
    seaDL.fire(result_sub_2)

    seaDL.fire(result_mul_1)
    seaDL.fire(result_mul_2)

    seaDL.fire(result_neg_1)
    seaDL.fire(result_neg_1)

    seaDL.fire(result_div_1)
    seaDL.fire(result_div_2)

    seaDL.fire(result_pow_1)
    seaDL.fire(result_pow_2)

    np.testing.assert_allclose(np.array(result_neg_1.data), np.array([-1.0, 2.0, -3.0]))   

    np.testing.assert_allclose(np.array(result_add_1.data), np.array([5.0, 3.0, 9.0]))
    np.testing.assert_allclose(np.array(result_add_2.data), np.array([3.0, 0.0, 5.0]))

    np.testing.assert_allclose(np.array(result_sub_1.data), np.array([-3.0, -7.0, -3.0]))
    np.testing.assert_allclose(np.array(result_sub_2.data), np.array([-2.0, -5.0, 0.0]))

    np.testing.assert_allclose(np.array(result_mul_1.data), np.array([4.0, -10.0, 18.0]))
    np.testing.assert_allclose(np.array(result_mul_2.data), np.array([20.0, 25.0, 30.0]))

    np.testing.assert_allclose(np.array(result_div_1.data), np.array([0.25, -0.4, 0.5]))
    np.testing.assert_allclose(np.array(result_div_2.data), np.array([0.8, 1.0, 1.2]))

    np.testing.assert_allclose(np.array(result_pow_1.data), np.array([1.0, 4.0, 9.0]))
    np.testing.assert_allclose(np.array(result_pow_2.data), np.array([1.0, -32.0, 729.0]))


def test_module(pytest_configure):
    net = SimpleNet()

    x = seaDL.Tensor([[0.5, 2.5, -3.4]])

    output = net(x)

    seaDL.fire(output)

    net.zero_grad()

    output.zero_grad()

    output.backward()

    np.testing.assert_allclose(np.array(output.data),
                               np.maximum(np.array(((x.data @ net.linear.weight.data.transpose()) + net.linear.bias.data)), 0),
                               rtol=1e-5)

    print(net.linear.bias.shape, net.linear.bias.grad.shape)

    assert gradient_check(output, net.linear.weight, h=1e-6, error_tolerance=0.03)
    assert gradient_check(output, net.linear.bias, h=1e-6, error_tolerance=0.03)


def test_auto_diff_1(pytest_configure):
    x = seaDL.Tensor([[[[2.0, 3.0, 4.0, 5.0], [-2.0, -3.0, -4.0, -5.0]]], [[[2.0, 3.0, 4.0, 5.0], [-2.0, -3.0, -4.0, -5.0]]]])

    w = seaDL.Tensor([[[[2.0, 3.0, 4.0, 5.0], [-2.0, -3.0, -4.0, -5.0]]], [[[2.0, 3.0, 4.0, 5.0], [-2.0, -3.0, -4.0, -5.0]]]], requires_grad=True)

    einsum_op = w.einsum("ijmn->j")

    seaDL.fire(einsum_op)

    einsum_op.zero_grad()

    einsum_op.backward()

    assert gradient_check(einsum_op, w, h=1e-6, error_tolerance=0.03)

    w.zero_grad()

    einsum_op = x.einsum("ijkm,ijkn->in", w)

    seaDL.fire(einsum_op)

    einsum_op.zero_grad()

    einsum_op.backward()

    assert gradient_check(einsum_op, w, h=1e-6, error_tolerance=0.06)


def test_auto_diff_2(pytest_configure):
    w = seaDL.Tensor([[[[2.0, 3.0, 4.0, 5.0], [-2.0, -3.0, -4.0, -5.0]]], [[[2.0, 3.0, 4.0, 5.0], [-2.0, -3.0, -4.0, -5.0]]]], requires_grad=True)

    sq_op = w.squeeze(dim=(1,))

    seaDL.fire(sq_op)

    sq_op.zero_grad()

    sq_op.backward()

    assert gradient_check(sq_op, w, h=1e-6, error_tolerance=0.03)

    w.zero_grad()

    unsq_op = w.unsqueeze(dim=(0,))

    seaDL.fire(unsq_op)

    unsq_op.zero_grad()

    unsq_op.backward()

    assert gradient_check(unsq_op, w, h=1e-6, error_tolerance=0.03)

    w.zero_grad()

    mean_op = w.mean(dim=(1, 2))

    seaDL.fire(mean_op)

    mean_op.zero_grad()

    mean_op.backward()

    assert gradient_check(mean_op, w, h=1e-6, error_tolerance=0.03)

    w.zero_grad()

    max_op = w.max(dim=(-3, -1))

    seaDL.fire(max_op)

    max_op.zero_grad()

    max_op.backward()

    assert gradient_check(max_op, w, h=1e-6, error_tolerance=0.03)


def test_auto_diff_3(pytest_configure):
    w = seaDL.Tensor([[[[2.0, 3.0, 4.0, 5.0], [-2.0, -3.0, -4.0, -5.0]]], [[[2.0, 3.0, 4.0, 5.0], [-2.0, -3.0, -4.0, -5.0]]]], requires_grad=True)

    a = seaDL.Tensor([[[[-2.0, 4.0, -1.0, 2.5], [2.0, -1.5, 4.0, 0.5]]], [[[-2.0, 7.0, 0.6, 2.1], [-3.0, 2.7, 1.2, 6.2]]]])

    c_actual = w.cat((a,), dim=1)

    seaDL.fire(c_actual)

    c_expected = seaDL.config.backend.concatenate(
        (seaDL.config.Array(w.data), seaDL.config.Array(a.data)),
        axis=1
    )

    np.testing.assert_allclose(np.array(c_actual.data), np.array(c_expected))

    c_actual.zero_grad()

    c_actual.backward()

    assert gradient_check(c_actual, w, h=1e-6, error_tolerance=0.03)

    w.zero_grad()

    c_actual.zero_grad()

    s_actual_a, _ = c_actual.split(2, dim=1)

    seaDL.fire(s_actual_a)

    expected_a, _ = seaDL.config.backend.split(seaDL.config.Array(c_actual.data), 2, axis=1)

    np.testing.assert_allclose(np.array(s_actual_a.data), np.array(expected_a))

    s_actual_a.operation.backward((seaDL.ones_like(s_actual_a).data, seaDL.zeros_like(s_actual_a).data))

    assert gradient_check(s_actual_a, w, h=1e-6, error_tolerance=0.03)

