from typing import Union, Optional, NamedTuple
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from collections import namedtuple
from functools import reduce
import random
import numpy as np
import torch
import mlx.core as mx

from ..base import Tensor

from graphviz import Digraph


@jaxtyped(typechecker=typechecker)
def prod(
    array: Union[tuple, list]
): 
    return reduce((lambda x, y: x * y), array)


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


def trace_graph(
        root: Tensor
) -> tuple[set, set]:
    """
    Trace computational graph
    """
    # nodes = Operator or Tensor
    # edges = dataflow
    nodes, dataflow = set(), set()

    def _build_graph(tensor: Tensor):
        if tensor not in nodes:
            nodes.add(tensor)

            operation = tensor.node

            if operation is not None:
                nodes.add(operation)

                # dataflow: Operation -> output Tensor
                dataflow.add((operation, tensor))

                # add a connection between Operation
                # and its input Tensors
                for input_tensor in operation.inputs:
                    # dataflow: input Tensor -> Operation
                    dataflow.add((input_tensor, operation))

                    _build_graph(input_tensor)

    _build_graph(root)

    return nodes, dataflow


def visualize_graph(
        root,
        format: str = 'svg',
        direction: str = 'TB'
):
    """
    Visualize computational graph

    Inputs:
        format: format to store the graph: 'svg', 'png'
        direction: direction for building the graph: ['TB', 'LR']
    """
    assert direction in ['TB', 'LR'], "direction has to be either 'TB' or 'LR'"

    nodes, dataflow = trace_graph(root)

    graph = Digraph(format=format, graph_attr={'rankdir': direction})

    for node in nodes:
        graph.node(name=str(id(node)), label=str(node), shape='record')

    for n1, n2 in dataflow:
        graph.edge(str(id(n1)), str(id(n2)))

    return graph

