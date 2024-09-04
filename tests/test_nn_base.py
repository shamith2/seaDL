import pytest
import mlx.core as mx

import seaML.nn as nn
from seaML.utils import visualize_graph


@pytest.fixture
def pytest_configure():
    pytest.device = mx.gpu
    pytest.n_tests = 10


def test_parameter(pytest_configure):
    param1 = nn.Parameter(mx.array([1.0, -2.0, 3.0]))
    param2 = nn.Parameter(mx.array([4.0, 5.0, 6.0]))

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

    assert mx.allclose(result_neg_1.fire().data, mx.array([-1.0, 2.0, -3.0])).all()    

    assert mx.allclose(result_add_1.fire().data, mx.array([5.0, 3.0, 9.0])).all()
    assert mx.allclose(result_add_2.fire().data, mx.array([3.0, 0.0, 5.0])).all()
    assert mx.allclose(result_add_3.fire().data, mx.array([7.0, 8.0, 9.0])).all()

    assert mx.allclose(result_sub_1.fire().data, mx.array([-3.0, -7.0, -3.0])).all()
    assert mx.allclose(result_sub_2.fire().data, mx.array([-2.0, -5.0, 0.0])).all()
    # assert mx.allclose(result_sub_3.fire().data, mx.array([-1.0, -2.0, -3.0])).all()

    assert mx.allclose(result_mul_1.fire().data, mx.array([4.0, -10.0, 18.0])).all()
    assert mx.allclose(result_mul_2.fire().data, mx.array([5.0, -10.0, 15.0])).all()
    assert mx.allclose(result_mul_3.fire().data, mx.array([-16.0, -20.0, -24.0])).all()

    assert mx.allclose(result_pow_1.fire().data, mx.array([1.0, 4.0, 9.0])).all()
    assert mx.allclose(result_pow_2.fire().data, mx.array([1.0, -32.0, 729.0])).all()


def test_module(pytest_configure):
    import sys

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.linear = nn.Linear(3, 2)
            self.linear.use_einsum = False

            self.relu = nn.ReLU()
        
        def __call__(self, x):
            out = self.linear(x)
            return self.relu(out)


    net = SimpleNet()

    x = nn.Tensor(mx.array([10.0, -2.0, 3.0]))

    output = net(x)

    c_output = output.fire()

    assert mx.allclose(c_output.data,
                       mx.maximum(((x.data @ net.linear.weight.data.transpose()) + net.linear.bias.data), 0.0)).all()

    # g = visualize_graph(c_output)

    # g.render('computational_graph', view=True, cleanup=True)

    output.backward()

    for name, param in net.named_parameters():
        print(name, param.grad)

    sys.exit()

