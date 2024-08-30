import mlx.core as mx
import pytest

from seaML.nn import BatchNorm2d


@pytest.fixture
def pytest_configure():
    pytest.device = mx.gpu
    pytest.n_tests = 10
    pytest.num_features = 3


def test_batchnorm2d_module(pytest_configure):
    bn = BatchNorm2d(pytest.num_features, device=pytest.device)

    params = bn.parameters()
    t_params = bn.trainable_parameters()

    assert bn.num_features == pytest.num_features

    assert isinstance(bn.weight, mx.array), f"weight has wrong type: {type(bn.weight)}"
    assert bn.weight.shape == params['weight'].shape

    assert isinstance(bn.bias, mx.array), f"bias has wrong type: {type(bn.bias)}"
    assert bn.bias.shape == params['bias'].shape

    assert isinstance(bn.running_mean, mx.array), f"running_mean has wrong type: {type(bn.running_mean)}"
    assert isinstance(bn.running_var, mx.array), f"running_var has wrong type: {type(bn.running_var)}"
    assert isinstance(bn.num_batches_tracked, mx.array), f"num_batches_tracked has wrong type: {type(bn.num_batches_tracked)}"

    assert 'running_mean' not in t_params.keys(), "running_mean should not be in list of trainable parameters since it is a buffer"
    assert 'running_var' not in t_params.keys(), "running_var should not be in list of trainable parameters since it is a buffer"
    assert 'num_batches_tracked' not in t_params.keys(), "num_batches_tracked should not be in list of trainable parameters since it is a buffer"


def test_batchnorm2d_forward(pytest_configure):
    '''
    For each channel, mean should be very close to 0 and std kinda close to 1 (because of eps)
    '''
    bn = BatchNorm2d(pytest.num_features, device=pytest.device)

    assert bn.training

    x = mx.random.normal(shape=(100, pytest.num_features, 3, 4))

    out = bn(x)
    assert x.shape == out.shape

    assert mx.allclose(mx.mean(out, axis=(0, 2, 3)), mx.zeros(pytest.num_features), atol=1e-5, rtol=1e-6).all()
    assert mx.allclose(mx.std(out, axis=(0, 2, 3)), mx.ones(pytest.num_features), atol=1e-5, rtol=1e-6).all()


def test_batchnorm2d_running_mean(pytest_configure):
    '''
    Over repeated forward calls with the same data in train mode,
    the running mean should converge to the actual mean
    '''
    bn = BatchNorm2d(pytest.num_features, momentum=0.6, device=pytest.device)
    assert bn.training

    x = mx.arange(12, dtype=mx.float32).reshape((2, 3, 2, 1))
    mean = mx.array([3.5000, 5.5000, 7.5000])
    num_batches = 30

    for i in range(num_batches):
        bn(x)
        expected_mean = (1 - (((1 - bn.momentum) ** (i + 1)))) * mean
        assert mx.allclose(bn.running_mean, expected_mean).all()

    assert bn.num_batches_tracked.item() == num_batches

    # Large enough momentum and num_batches -> running_mean should be very close to actual mean
    bn.eval()
    actual_eval_mean = mx.mean(bn(x), axis=(0, 2, 3))
    assert mx.allclose(actual_eval_mean, mx.zeros(3), atol=1e-7, rtol=1e-6).all()

