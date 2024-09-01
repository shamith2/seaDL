from typing import Union, Optional, NamedTuple
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from collections import namedtuple
import random
import numpy as np
import torch
import mlx.core as mx


@jaxtyped(typechecker=typechecker)
def get_strides(
        shape: tuple,
        device: mx.DeviceType
):
    """
    Like torch.Tensor.stride

    If shape of tensor is (2, 16, 32),
    then, the stride in dim 0 = 1 (since the elements in dim 0 are consecutive in memory),
    dim 1 = 32 (since elements in dim 1 are 32 elements apart) and
    dim 2 = 32 * 16 (since elements in dim 2 are 16 blocks apart where each block is 32 elements),
    so function will return (512, 32, 1)
    """
    strides = mx.array([1] * len(shape))

    strides[:-1] = mx.cumprod(mx.array(shape[::-1]), stream=device)[::-1][1:]

    # strides = mx.concatenate((strides, mx.array([1])), axis=0, stream=device)

    return strides


@jaxtyped(typechecker=typechecker)
def _pair_value(
        v: Union[int, tuple[int, int]]
) -> tuple[int, int]:
    '''
    Convert v to a pair of int, if it isn't already
    '''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return v

    elif isinstance(v, int):
        return (v, v)

    raise ValueError(v)


@jaxtyped(typechecker=typechecker)
def convert_and_shuffle_dataset(
        X: Union[np.ndarray, torch.Tensor, mx.array],
        y: Union[np.ndarray, torch.Tensor, mx.array],
        shuffle: bool,
        seed: Optional[int] = None
) -> tuple[tuple, tuple]:
    if isinstance(X, torch.Tensor):
        X = mx.array(X.detach().numpy())

    elif isinstance(X, np.ndarray):
        X = mx.array(X)

    else:
        pass

    if isinstance(y, torch.Tensor):
        y = mx.array(y.detach().numpy())

    elif isinstance(X, np.ndarray):
        y = mx.array(y)

    else:
        pass

    dataset = list(zip(X, y))

    if shuffle:
        random.seed(seed)
        random.shuffle(dataset)

    tuple_X, tuple_y = zip(*dataset)

    return tuple_X, tuple_y


@jaxtyped(typechecker=typechecker)
def train_test_split(
        X: Union[np.ndarray, torch.Tensor, mx.array],
        y: Union[np.ndarray, torch.Tensor, mx.array],
        train_val_split: Union[tuple, float],
        shuffle: bool,
        seed: Optional[int] = None
) -> NamedTuple:
    # error checking
    if isinstance(train_val_split, (tuple, list)) and len(train_val_split) != 2:
        raise ValueError("size of train_val_split should be 2 if train_val_split is a tuple")

    if isinstance(train_val_split, float):
        do_validation_split = False

        training_split = train_val_split
        test_split = 1.0 - training_split

        if training_split <= 0.0 or test_split <= 0.0:
            raise ValueError("check train_val_split values")

    else:
        do_validation_split = True

        training_split = train_val_split[0]
        validation_split = train_val_split[1]
        test_split = 1.0 - (training_split + validation_split)

        if training_split <= 0.0 or validation_split <= 0.0 or test_split <= 0.0:
            raise ValueError("check train_val_split values")

    dataset_size = len(X)
    train_split_idx = int(training_split * dataset_size)

    if do_validation_split:
        validation_split_idx = int(validation_split * dataset_size)

    X, y = convert_and_shuffle_dataset(X, y, shuffle, seed)

    # training data
    X_train, y_train = X[:train_split_idx], y[:train_split_idx]

    # validation data
    if do_validation_split:
        X_validation = X[train_split_idx:(train_split_idx + validation_split_idx)]
        y_validation = y[train_split_idx:(train_split_idx + validation_split_idx)]

        # test data
        X_test = X[(train_split_idx + validation_split_idx):]
        y_test = y[(train_split_idx + validation_split_idx):]

    else:
        X_test, y_test = X[train_split_idx:], y[train_split_idx:]

    if do_validation_split:
        dataSplit = namedtuple("dataSplit",
                               "X_train y_train X_validation y_validation X_test y_test")

        return_tuple = dataSplit(X_train, y_train, X_validation, y_validation, X_test, y_test)

    else:
        dataSplit = namedtuple("dataSplit",
                               "X_train y_train X_test y_test")

        return_tuple = dataSplit(X_train, y_train, X_test, y_test)

    return return_tuple


@jaxtyped(typechecker=typechecker)
class DataLoader:
    def __init__(
            self,
            X: Union[tuple, np.ndarray, torch.Tensor, mx.array],
            y: Union[tuple, np.ndarray, torch.Tensor, mx.array],
            batch_size: int,
            shuffle: bool,
            seed: Optional[int] = None,
            device: Optional[mx.DeviceType] = None
    ):
        self.device = mx.cpu if not device else device
        self.seed = seed

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.X, self.y = X, y

        # could be a tuple, numpy array or numpy scalar
        if isinstance(self.X, (tuple, list)) or type(self.X).__module__ == np.__name__:
            self.X = mx.array(self.X)

        # could be a tuple, numpy array or numpy scalar
        if isinstance(self.y, (tuple, list)) or type(self.y).__module__ == np.__name__:
            self.y = mx.array(self.y)

        self._convert_and_shuffle_dataset()

        self.dataset_size = len(self.y) # len(self.x) can also be used
        self.mini_batch_dataset_size = self.dataset_size // self.batch_size

    def _convert_and_shuffle_dataset(self):
        self.X, self.y = convert_and_shuffle_dataset(
                self.X,
                self.y,
                self.shuffle,
                self.seed
        )

    def __len__(self):
        return self.mini_batch_dataset_size

    def __iter__(self):
        for batch_idx in range(0, self.dataset_size, self.batch_size):
            yield (mx.stack(self.X[batch_idx:(batch_idx + self.batch_size)], axis=0, stream=self.device),
                   mx.stack(self.y[batch_idx:(batch_idx + self.batch_size)], axis=0, stream=self.device))

