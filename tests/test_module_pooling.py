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


# def test_maxpool2d_module(MaxPool2d, n_tests=20, tuples=False):
#     import numpy as np
#     for i in range(n_tests):
#         b = np.random.randint(1, 10)
#         h = np.random.randint(10, 50)
#         w = np.random.randint(10, 50)
#         ci = np.random.randint(1, 20)
#         if tuples:
#             stride = None if np.random.random() < 0.5 else tuple(np.random.randint(1, 5, size=(2,)))
#             kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
#             kH, kW = kernel_size
#             padding = np.random.randint(0, 1 + kH // 2), np.random.randint(0, 1 + kW // 2)
#         else:
#             stride = None if np.random.random() < 0.5 else np.random.randint(1, 5)
#             kernel_size = np.random.randint(1, 10)
#             padding = np.random.randint(0, 1 + kernel_size // 2)
#         x = t.randn((b, ci, h, w))
#         my_output = MaxPool2d(
#             kernel_size,
#             stride=stride,
#             padding=padding,
#         )(x)

#         torch_output = nn.MaxPool2d(
#             kernel_size,
#             stride=stride,
#             padding=padding,
#         )(x)
#         t.testing.assert_close(my_output, torch_output)
#     print("All tests in `test_maxpool2d_module` passed!")

# def test_conv2d_module(Conv2d, n_tests=5, tuples=False):
#     '''
#     Your weight should be called 'weight' and have an appropriate number of elements.
#     '''
#     m = Conv2d(4, 5, 3)
#     assert isinstance(m.weight, t.nn.parameter.Parameter), "Weight should be registered a parameter!"
#     assert m.weight.nelement() == 4 * 5 * 3 * 3
#     import numpy as np
#     for i in range(n_tests):
#         b = np.random.randint(1, 10)
#         h = np.random.randint(10, 300)
#         w = np.random.randint(10, 300)
#         ci = np.random.randint(1, 20)
#         co = np.random.randint(1, 20)
#         if tuples:
#             stride = tuple(np.random.randint(1, 5, size=(2,)))
#             padding = tuple(np.random.randint(0, 5, size=(2,)))
#             kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
#         else:
#             stride = np.random.randint(1, 5)
#             padding = np.random.randint(0, 5)
#             kernel_size = np.random.randint(1, 10)
#         x = t.randn((b, ci, h, w))
#         my_conv = Conv2d(in_channels=ci, out_channels=co, kernel_size=kernel_size, stride=stride, padding=padding)
#         my_output = my_conv(x)
#         torch_output = t.conv2d(x, my_conv.weight, stride=stride, padding=padding)
#         t.testing.assert_close(my_output, torch_output)
#     print("All tests in `test_conv2d_module` passed!")

# def test_relu(ReLU):
#     x = t.randn(10) - 0.5
#     actual = ReLU()(x)
#     expected = F.relu(x)
#     t.testing.assert_close(actual, expected)
#     print("All tests in `test_relu` passed!")

# def test_flatten(Flatten):
#     x = t.arange(24).reshape((2, 3, 4))
#     assert Flatten(start_dim=0)(x).shape == (24,)
#     assert Flatten(start_dim=1)(x).shape == (2, 12)
#     assert Flatten(start_dim=0, end_dim=1)(x).shape == (6, 4)
#     assert Flatten(start_dim=0, end_dim=-2)(x).shape == (6, 4)
#     print("All tests in `test_flatten` passed!")

# def test_linear_forward(Linear):
#     '''Your Linear should produce identical results to torch.nn given identical parameters.'''
#     x = t.rand((10, 512))
#     yours = Linear(512, 64)
#     assert yours.weight.shape == (64, 512), f"Linear layer weights have wrong shape: {yours.weight.shape}, expected shape = (64, 512)"
#     assert yours.bias.shape == (64,), f"Linear layer bias has wrong shape: {yours.bias.shape}, expected shape = (64,)"
#     official = t.nn.Linear(512, 64)
#     yours.weight = official.weight
#     yours.bias = official.bias
#     actual = yours(x)
#     expected = official(x)
#     t.testing.assert_close(actual, expected)
#     print("All tests in `test_linear_forward` passed!")

# def test_linear_parameters(Linear):
#     m = Linear(2, 3)
#     params = dict(m.named_parameters())
#     assert len(params) == 2, f"Your model has {len(params)} recognized Parameters"
#     assert set(params.keys()) == {"weight", "bias"}, f"For compatibility with PyTorch, your fields should be named weight and bias, not {tuple(params.keys())}"
#     print("All tests in `test_linear_parameters` passed!")

# def test_linear_no_bias(Linear):
    
#     x = t.rand((10, 512))
#     yours = Linear(512, 64, bias=False)

#     assert yours.bias is None, "Bias should be None when not enabled."
#     assert len(list(yours.parameters())) == 1

#     official = nn.Linear(512, 64, bias=False)
#     yours.weight = official.weight
#     actual = yours(x)
#     expected = official(x)
#     t.testing.assert_close(actual, expected)
#     print("All tests in `test_linear_no_bias` passed!")

# def test_mlp(SimpleMLP):
#     mlp: nn.Module = SimpleMLP()
#     num_params = sum(p.numel() for p in mlp.parameters())
#     assert num_params == 79510, f"Expected (28*28 + 1) * 100 + ((100 + 1) * 10) = 79510 parameters, got {num_params}"
#     # Get list of all the modules which aren't the the top-level module, or the Sequential module
#     inner_modules = [m for m in mlp.modules() if m is not mlp and not isinstance(m, nn.Sequential)]
#     assert len(inner_modules) == 4, f"Expected 4 modules (flatten, linear1, relu, linear2), got {len(inner_modules)}"
#     print("All tests in `test_mlp` passed!")

# def test_batchnorm2d_module(BatchNorm2d):
#     '''The public API of the module should be the same as the real PyTorch version.'''
#     num_features = 2
#     bn = BatchNorm2d(num_features)
#     assert bn.num_features == num_features
#     assert isinstance(bn.weight, t.nn.parameter.Parameter), f"weight has wrong type: {type(bn.weight)}"
#     assert isinstance(bn.bias, t.nn.parameter.Parameter), f"bias has wrong type: {type(bn.bias)}"
#     assert isinstance(bn.running_mean, t.Tensor), f"running_mean has wrong type: {type(bn.running_mean)}"
#     assert isinstance(bn.running_var, t.Tensor), f"running_var has wrong type: {type(bn.running_var)}"
#     assert isinstance(bn.num_batches_tracked, t.Tensor), f"num_batches_tracked has wrong type: {type(bn.num_batches_tracked)}"
#     print("All tests in `test_batchnorm2d_module` passed!")

# def test_batchnorm2d_forward(BatchNorm2d):
#     '''For each channel, mean should be very close to 0 and std kinda close to 1 (because of eps).'''
#     num_features = 2
#     bn = BatchNorm2d(num_features)
#     assert bn.training
#     x = t.randn((100, num_features, 3, 4))
#     out = bn(x)
#     assert x.shape == out.shape
#     t.testing.assert_close(out.mean(dim=(0, 2, 3)), t.zeros(num_features))
#     t.testing.assert_close(out.std(dim=(0, 2, 3)), t.ones(num_features), atol=1e-3, rtol=1e-3)
#     print("All tests in `test_batchnorm2d_forward` passed!")

# def test_batchnorm2d_running_mean(BatchNorm2d):
#     '''Over repeated forward calls with the same data in train mode, the running mean should converge to the actual mean.'''
#     bn = BatchNorm2d(3, momentum=0.6)
#     assert bn.training
#     x = t.arange(12).float().view((2, 3, 2, 1))
#     mean = t.tensor([3.5000, 5.5000, 7.5000])
#     num_batches = 30
#     for i in range(num_batches):
#         bn(x)
#         expected_mean = (1 - (((1 - bn.momentum) ** (i + 1)))) * mean
#         t.testing.assert_close(bn.running_mean, expected_mean)
#     assert bn.num_batches_tracked.item() == num_batches

#     # Large enough momentum and num_batches -> running_mean should be very close to actual mean
#     bn.eval()
#     actual_eval_mean = bn(x).mean((0, 2, 3))
#     t.testing.assert_close(actual_eval_mean, t.zeros(3))
#     print("All tests in `test_batchnorm2d_running_mean` passed!")
    
# def test_averagepool(AveragePool):
#     x = t.arange(24).reshape((1, 2, 3, 4)).float()
#     actual = AveragePool()(x)
#     expected = t.tensor([[5.5, 17.5]])
#     t.testing.assert_close(actual, expected)
#     print("All tests in `test_averagepool` passed!")

# def test_residual_block(ResidualBlock):
#     '''
#     Test the user's implementation of `ResidualBlock`.
#     '''
#     import part2_cnns.solutions as solutions

#     # Create random input tensor
#     x = t.randn(1, 3, 64, 64)

#     # Instantiate both user and reference models
#     user_model = ResidualBlock(in_feats=3, out_feats=3)
#     ref_model = solutions.ResidualBlock(in_feats=3, out_feats=3)
#     # Check parameter count
#     user_params = sum(p.numel() for p in user_model.parameters())
#     ref_params = sum(p.numel() for p in ref_model.parameters())
#     # Special case this scenario because occasionally people will
#     # unconditionally create the right-hand branch and then only conditionally
#     # switch in the forward method whether to use the right-hand branch or not.
#     if user_params > ref_params:
#         error_message = f"""
#         When the first_stride=1, there are more parameters ({user_params}) than
#         expected ({ref_params}). Make sure that you don't create unnecessary
#         convolutions for the right-hand branch when first_stride=1. That is your
#         initialization code should only initialize the right-hand branch when
#         first_stride is not 1.
#         """
#         raise AssertionError(error_message)
#     assert user_params == ref_params, f"Parameter count mismatch (when first_stride=1). Expected {ref_params}, got {user_params}."
#     # Check forward function output is correct shape
#     user_output = user_model(x)
#     assert user_output.shape == (1, 3, 64, 64), f"Incorrect shape, expected (batch=1, out_feats=4, height=64, width=64), got {user_output.shape}"
#     print("Passed all tests when first_stride=1")
    
#     # Same checks, but now with nontrivial stride
#     user_model = ResidualBlock(in_feats=3, out_feats=4, first_stride=2)
#     ref_model = solutions.ResidualBlock(in_feats=3, out_feats=4, first_stride=2)
#     user_params = sum(p.numel() for p in user_model.parameters())
#     ref_params = sum(p.numel() for p in ref_model.parameters())
#     assert user_params == ref_params, f"Parameter count mismatch (when first_stride>1). Expected {ref_params}, got {user_params}."
#     user_output = user_model(x)
#     assert user_output.shape == (1, 4, 32, 32), f"Incorrect shape, expected (batch=1, out_feats=4, height/first_stride=32, width/first_stride=32), got {user_output.shape}"
#     print("Passed all tests when first_stride>1")

#     print("All tests in `test_residual_block` passed!")

# def test_block_group(BlockGroup):
#     '''
#     Test the user's implementation of `ResidualBlock`.
#     '''
#     import part2_cnns.solutions as solutions

#     # Create random input tensor
#     x = t.randn(1, 3, 64, 64)

#     # Instantiate both user and reference models
#     user_model = BlockGroup(n_blocks=2, in_feats=3, out_feats=3)
#     ref_model = solutions.BlockGroup(n_blocks=2, in_feats=3, out_feats=3)
#     # Check parameter count
#     user_params = sum(p.numel() for p in user_model.parameters())
#     ref_params = sum(p.numel() for p in ref_model.parameters())
#     assert user_params == ref_params, "Parameter count mismatch (when n_blocks=2, first_stride=1)"
#     # Check forward function output is correct shape
#     user_output = user_model(x)
#     assert user_output.shape == (1, 3, 64, 64), f"Incorrect shape, expected (batch=1, out_feats=4, height=64, width=64), got {user_output.shape}"
#     print("Passed all tests when first_stride=1")
    
#     # Same checks, but now with nontrivial stride
#     user_model = BlockGroup(n_blocks=2, in_feats=3, out_feats=4, first_stride=2)
#     ref_model = solutions.BlockGroup(n_blocks=2, in_feats=3, out_feats=4, first_stride=2)
#     user_params = sum(p.numel() for p in user_model.parameters())
#     ref_params = sum(p.numel() for p in ref_model.parameters())
#     assert user_params == ref_params, "Parameter count mismatch (when n_blocks=2, first_stride>1)"
#     user_output = user_model(x)
#     assert user_output.shape == (1, 4, 32, 32), f"Incorrect shape, expected (batch=1, out_feats=4, height/first_stride=32, width/first_stride=32), got {user_output.shape}"
#     print("Passed all tests when first_stride>1")

    
#     # Same checks, but now with a larger n_blocks
#     user_model = BlockGroup(n_blocks=5, in_feats=3, out_feats=4, first_stride=2)
#     ref_model = solutions.BlockGroup(n_blocks=5, in_feats=3, out_feats=4, first_stride=2)
#     user_params = sum(p.numel() for p in user_model.parameters())
#     ref_params = sum(p.numel() for p in ref_model.parameters())
#     assert user_params == ref_params, "Parameter count mismatch (when n_blocks=5, first_stride>1)"
#     user_output = user_model(x)
#     assert user_output.shape == (1, 4, 32, 32), f"Incorrect shape, expected (batch=1, out_feats=4, height/first_stride=32, width/first_stride=32), got {user_output.shape}"
#     print("Passed all tests when n_blocks>2")

#     print("All tests in `test_block_group` passed!")


# def test_get_resnet_for_feature_extraction(get_resnet_for_feature_extraction):

#     resnet: nn.Module = get_resnet_for_feature_extraction(10)

#     num_params = len(list(resnet.parameters()))

#     error_msg = "\nNote - make sure you've defined your resnet modules in the correct order (with the final linear layer last), \
# otherwise this can cause issues for the test function."

#     # Check all gradients are correct
#     for i, (name, param) in enumerate(resnet.named_parameters()):
#         if i < num_params - 2:
#             assert not param.requires_grad, f"Found param {name!r} before the final layer, which has requires_grad=True." + error_msg
#         else:
#             assert param.requires_grad, f"Found param {name!r} in the final layer, which has requires_grad=False." + error_msg
#             if param.ndim == 2:
#                 assert tuple(param.shape) == (10, 512), f"Expected final linear layer weights to have shape (n_classes=10, 512), instead found {tuple(param.shape)}" + error_msg
#             else:
#                 assert tuple(param.shape) == (10,), f"Expected final linear layer bias to have shape (n_classes=10,), instead found {tuple(param.shape)}" + error_msg
    
#     print("All tests in `test_get_resnet_for_feature_extraction` passed!")
