import pytest
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from torch.utils.data import DataLoader as torchDataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split as sk_train_test_split

import seaDL
from seaDL.nn import Parameter, Module, Sequential
from seaDL.optim import SGD
from seaDL.utils import DataLoader as DataLoader
from seaDL.utils import train_test_split


@pytest.fixture
def pytest_configure():
    pytest.device = None
    pytest.n_tests = 10
    pytest.seed = 4224


class Net_torch(torch.nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int
    ):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim, bias=True)
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        return self.layers(x)


class Net(Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int
    ):
        super().__init__()

        self.layers = Sequential(
            OrderedDict([
                ('l1', seaDL.nn.Linear(in_dim, hidden_dim, bias=True)),
                ('r1', seaDL.nn.ReLU()),
                ('l2', seaDL.nn.Linear(hidden_dim, hidden_dim, bias=True)),
                ('r2', seaDL.nn.ReLU()),
                ('l3', seaDL.nn.Linear(in_dim, out_dim, bias=True))
            ])
        )

    def __call__(
            self,
            x: seaDL.Tensor
    ) -> seaDL.Tensor:
        return self.layers(x)


def get_moon_data():
    X, y = make_moons(n_samples=512, noise=0.05, random_state=pytest.seed)

    X = X.astype('float32')
    y = y.astype('int64')

    return DataLoader(X, y, batch_size=128, shuffle=False)


def train_with_optim(model_t, model, optimizer_t, optim, dl):
    for idx, (x_i, y_i) in enumerate(dl):
        xt_i = torch.from_numpy(np.array(x_i.data))
        yt_i = torch.from_numpy(np.array(y_i.data))

        optimizer_t.zero_grad()
        optim.zero_grad()

        out_t = model_t(xt_i)
        out = model(x_i)

        loss_t = F.nll_loss(out_t - out_t.sum(dim=-1, keepdim=True), yt_i, reduction='mean')
        loss = seaDL.nn.functional.nll_loss(out - out.sum(dim=-1, keepdim=True), y_i, reduction='mean')

        if seaDL.config.is_backend_numpy():
            loss = loss.squeeze(dim=0)

        seaDL.fire(loss)

        print(loss_t, loss_t.shape)
        print(loss, out.shape)

        # g = seaDL.utils.visualize_graph(loss)
        # g.render('computational_graph_2', view=True, cleanup=True)

        torch.testing.assert_close(
            out_t,
            torch.from_numpy(np.array(out.data)),
            rtol=1e-5,
            atol=1e-5
        )

        torch.testing.assert_close(
            loss_t,
            torch.from_numpy(np.array(loss.data)),
            rtol=1e-5,
            atol=1e-5
        )

        loss_t.backward()
        loss.backward()

        with torch.inference_mode():
            optimizer_t.step()

        optim.step()


def test_train_test_split(pytest_configure):
    X, y = make_moons(n_samples=512, noise=0.05, random_state=pytest.seed)

    X_train, X_test, y_train, y_test = sk_train_test_split(
        X, y, train_size=0.8, shuffle=False
    )

    dataSplit = train_test_split(X, y, train_val_split=0.8, shuffle=False)

    assert np.allclose(np.array(X_train), np.array(dataSplit.X_train.data))
    assert np.allclose(np.array(X_test), np.array(dataSplit.X_test.data))
    assert np.allclose(np.array(y_train), np.array(dataSplit.y_train.data))
    assert np.allclose(np.array(y_test), np.array(dataSplit.y_test.data))

    dataset_size = len(X)

    assert len(X) == len(y)

    dataSplit = train_test_split(X, y, train_val_split=0.8, shuffle=True)

    assert len(dataSplit.X_train.data) == int(0.8 * dataset_size)
    assert len(dataSplit.y_train.data) == int(0.8 * dataset_size)

    assert len(dataSplit.X_test.data) == dataset_size - int(0.8 * dataset_size)
    assert len(dataSplit.y_test.data) == dataset_size - int(0.8 * dataset_size)

    dataSplit = train_test_split(X, y, train_val_split=(0.8, 0.1), shuffle=True)

    assert len(dataSplit.X_train.data) == int(0.8 * dataset_size)
    assert len(dataSplit.y_train.data) == int(0.8 * dataset_size)

    assert len(dataSplit.X_validation.data) == int(0.1 * dataset_size)
    assert len(dataSplit.y_validation.data) == int(0.1 * dataset_size)

    assert len(dataSplit.X_test.data) == dataset_size - int(0.9 * dataset_size)
    assert len(dataSplit.y_test.data) == dataset_size - int(0.9 * dataset_size)


def test_dataloader(pytest_configure):
    X, y = make_moons(n_samples=512, noise=0.05, random_state=pytest.seed)

    X_torch = torch.tensor(X, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.int64)

    dl_torch = torchDataLoader(TensorDataset(X_torch, y_torch), batch_size=128, shuffle=False)
    my_dl = DataLoader(X, y, batch_size=128, shuffle=False)

    store_xi = []
    store_yi = []

    for x_i, y_i in dl_torch:
        store_xi.append(x_i.detach().numpy())
        store_yi.append(y_i.detach().numpy())

    for i, (x_i, y_i) in enumerate(my_dl):
        assert isinstance(x_i, seaDL.Tensor)
        assert isinstance(y_i, seaDL.Tensor)

        assert np.allclose(store_xi[i], np.array(x_i.data))
        assert np.allclose(store_yi[i], np.array(y_i.data))


def test_optim_sgd(pytest_configure):
    test_cases = [
        dict(lr=0.1, momentum=0.0, weight_decay=0.0),
        dict(lr=0.1, momentum=0.7, weight_decay=0.1, dampening=0.1),
        dict(lr=0.1, momentum=0.5, weight_decay=0.1),
        dict(lr=0.1, momentum=0.5, weight_decay=0.05, dampening=0.2),
        dict(lr=0.2, momentum=0.8, weight_decay=0.05),
    ]

    my_dl = get_moon_data()

    for opt_config in test_cases:
        torch.manual_seed(pytest.seed)
        # seaDL.config.backend.random.seed(pytest.seed)

        model_torch = Net_torch(2, 32, 2)
        model = Net(2, 32, 2)

        model.layers[0].weight = Parameter(seaDL.Tensor(model_torch.layers[0].weight.detach().numpy()))
        model.layers[0].bias = Parameter(seaDL.Tensor(model_torch.layers[0].bias.detach().numpy()))
        model.layers[2].weight = Parameter(seaDL.Tensor(model_torch.layers[2].weight.detach().numpy()))
        model.layers[2].bias = Parameter(seaDL.Tensor(model_torch.layers[2].bias.detach().numpy()))
        model.layers[4].weight = Parameter(seaDL.Tensor(model_torch.layers[4].weight.detach().numpy()))
        model.layers[4].bias = Parameter(seaDL.Tensor(model_torch.layers[4].bias.detach().numpy()))

        optim_torch = torch.optim.SGD(model_torch.parameters(), **opt_config)
        optim = SGD(model.parameters(), **opt_config)

        train_with_optim(model_torch, model, optim_torch, optim, my_dl)

        w0_correct = model_torch.layers[0].weight
        w1_correct = model_torch.layers[2].weight
        w2_correct = model_torch.layers[4].weight

        b0_correct = model_torch.layers[0].bias
        b1_correct = model_torch.layers[2].bias
        b2_correct = model_torch.layers[4].bias

        w0_submitted = torch.from_numpy(np.array(model.layers[0].weight.data))
        w1_submitted = torch.from_numpy(np.array(model.layers[2].weight.data))
        w2_submitted = torch.from_numpy(np.array(model.layers[4].weight.data))

        b0_submitted = torch.from_numpy(np.array(model.layers[0].bias.data))
        b1_submitted = torch.from_numpy(np.array(model.layers[2].bias.data))
        b2_submitted = torch.from_numpy(np.array(model.layers[4].bias.data))

        assert isinstance(w0_correct, torch.Tensor)
        assert isinstance(model.layers[0].weight, Parameter)

        torch.testing.assert_close(w0_correct, w0_submitted, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(w1_correct, w1_submitted, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(w2_correct, w2_submitted, rtol=1e-5, atol=1e-5)

        torch.testing.assert_close(b0_correct, b0_submitted, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(b1_correct, b1_submitted, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(b2_correct, b2_submitted, rtol=1e-5, atol=1e-5)

