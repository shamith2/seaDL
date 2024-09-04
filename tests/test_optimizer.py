import torch
import torch.nn.functional as F
import numpy as np
import pytest

from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split as sk_train_test_split

import mlx.core as mx

import seaML.nn as nn
from seaML.nn import Module
from seaML.optim import SGD
from seaML.utils import DataLoader as mxDataLoader
from seaML.utils import train_test_split


@pytest.fixture
def pytest_configure():
    pytest.device = mx.gpu
    pytest.n_tests = 10


class Net_torch(torch.nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int
    ):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        return self.layers(x)


class Net_mx(Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int
    ):
        super().__init__()

        self.layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        ]

    def forward(
            self,
            x: mx.array
    ) -> mx.array:
        for l in self.layers:
            out = l(x)

        return out


def get_moon_data():
    X, y = make_moons(n_samples=512, noise=0.05, random_state=354)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)

    return DataLoader(TensorDataset(X, y), batch_size=128, shuffle=False)


def train_with_optim(model, optimizer):
    dl = get_moon_data()

    for x_i, y_i in dl:
        optimizer.zero_grad()

        loss = F.cross_entropy(model(x_i), y_i)
        loss.backward()

        optimizer.step()


def test_train_test_split(pytest_configure):
    X, y = make_moons(n_samples=512, noise=0.05, random_state=42)

    X_train, X_test, y_train, y_test = sk_train_test_split(
        X, y, train_size=0.8, shuffle=False
    )

    dataSplit = train_test_split(X, y, train_val_split=0.8, shuffle=False)

    assert np.allclose(np.array(X_train), np.array(dataSplit.X_train))
    assert np.allclose(np.array(X_test), np.array(dataSplit.X_test))
    assert np.allclose(np.array(y_train), np.array(dataSplit.y_train))
    assert np.allclose(np.array(y_test), np.array(dataSplit.y_test))

    dataset_size = len(X)

    assert len(X) == len(y)

    dataSplit = train_test_split(X, y, train_val_split=0.8, shuffle=True)

    assert len(dataSplit.X_train) == int(0.8 * dataset_size)
    assert len(dataSplit.y_train) == int(0.8 * dataset_size)

    assert len(dataSplit.X_test) == dataset_size - int(0.8 * dataset_size)
    assert len(dataSplit.y_test) == dataset_size - int(0.8 * dataset_size)

    dataSplit = train_test_split(X, y, train_val_split=(0.8, 0.1), shuffle=True)

    assert len(dataSplit.X_train) == int(0.8 * dataset_size)
    assert len(dataSplit.y_train) == int(0.8 * dataset_size)

    assert len(dataSplit.X_validation) == int(0.1 * dataset_size)
    assert len(dataSplit.y_validation) == int(0.1 * dataset_size)

    assert len(dataSplit.X_test) == dataset_size - int(0.9 * dataset_size)
    assert len(dataSplit.y_test) == dataset_size - int(0.9 * dataset_size)


def test_dataloader(pytest_configure):
    X, y = make_moons(n_samples=512, noise=0.05, random_state=42)

    X_torch = torch.tensor(X, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.int64)

    dl_torch = DataLoader(TensorDataset(X_torch, y_torch), batch_size=128, shuffle=False)
    my_dl = mxDataLoader(X, y, batch_size=128, shuffle=False)

    store_xi = []
    store_yi = []

    for x_i, y_i in dl_torch:
        store_xi.append(x_i.detach().numpy())
        store_yi.append(y_i.detach().numpy())

    for i, (x_i, y_i) in enumerate(my_dl):
        assert isinstance(x_i, mx.array)
        assert isinstance(y_i, mx.array)

        assert np.allclose(store_xi[i], np.array(x_i))
        assert np.allclose(store_yi[i], np.array(y_i))


# def test_optim_sgd(pytest_configure):
#     test_cases = [
#         dict(lr=0.1, momentum=0.0, weight_decay=0.0),
#         dict(lr=0.1, momentum=0.7, weight_decay=0.1, dampening=0.1),
#         dict(lr=0.1, momentum=0.5, weight_decay=0.1),
#         dict(lr=0.1, momentum=0.5, weight_decay=0.05, dampening=0.2),
#         dict(lr=0.2, momentum=0.8, weight_decay=0.05),
#     ]

#     for opt_config in test_cases:
#         torch.manual_seed(42)
#         mx.random.seed(42)

#         model_torch = Net_torch(2, 32, 2)
#         optim_torch = torch.optim.SGD(model_torch.parameters(), **opt_config)
#         train_with_optim(model_torch, optim_torch)
#         w0_correct = model_torch.layers[0].weight

#         model_mx = Net_mx(2, 32, 2)
#         optim_mx = SGD(model_mx, **opt_config)
#         train_with_optim(model_mx, optim_mx)
#         w0_submitted = model_mx.layers[0].weight

#         assert isinstance(w0_correct, torch.Tensor)
#         assert isinstance(w0_submitted, mx.array)

#         w0_submitted = np.array(w0_submitted)

#         torch.testing.assert_close(w0_correct, torch.from_numpy(w0_submitted), rtol=0, atol=1e-5)

