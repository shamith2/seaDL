from typing import Union, Optional, Iterable, NamedTuple
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

import copy
from collections import namedtuple
import itertools
import random

from seaDL import config, ArrayType
from seaDL import Tensor, fire, zeros_like

from graphviz import Digraph

import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


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

            operation = tensor.operation

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
        graph.node(name=str(id(node)), label=node.sprint(), shape='record')

    for n1, n2 in dataflow:
        graph.edge(str(id(n1)), str(id(n2)))

    return graph


@jaxtyped(typechecker=typechecker)
def reset_graph(
    root: Tensor,
    reference_tensor: Optional[Tensor] = None
):
    """
    2 scenarios to set operation.fired = False
        (1) if ref_tensor is a input to an operator
            but not an output to any operator (leaf tensor)
        (2) if ref_tensor is an output to an operator
            but not a input to any operator
        (3) if ref_tensor is a input to some operator and output
            to another operator, it should be detected in (1) or (2)
    """
    nodes = set()

    def _reset_graph(tensor: Tensor):
        if tensor not in nodes:
            nodes.add(tensor)

            operation = tensor.operation

            if operation is not None:
                operation.fired = False

                op_inputs = operation.inputs

                # scenario (1) and (2) detection
                if ((reference_tensor not in op_inputs)
                    and (id(tensor) != id(reference_tensor))):
                    for node_input in op_inputs:
                        _reset_graph(node_input)

    _reset_graph(root)


def gradient_check(
        root: Tensor,
        reference_tensor: Tensor,
        h: float = 1e-6,
        error_tolerance: float = 0.05,
        strict: Optional[bool] = False
) -> bool:
    """
    Return True if analytical gradient and numerical gradient are close

    Compute gradient of function fn at point p

    gradient = (fn(p + h) - fn(p - h)) / (2 * h) as h -> 0

    Inputs:
        root: Tensor: output tensor for the computational graph
        reference_tensor: the tensor for doing the gradient check
        h: float = Limit h -> 0

    Outputs:
        None = Raise AssertionError if analytics gradient and numerical gradient are not close
    """
    if not reference_tensor.requires_grad:
        raise ValueError("requires_grad for reference_tensor is False")

    analytical_gradient = copy.deepcopy(reference_tensor.grad)
    numerical_gradient = zeros_like(reference_tensor, requires_grad=False)

    shape = reference_tensor.data.shape

    for idx in itertools.product(*(range(s) for s in shape)):
        original_value: ArrayType = copy.deepcopy(reference_tensor.data)

        # turn off fired for operations until reference_tensor
        reset_graph(root, reference_tensor)

        # pertube one element at a time
        reference_tensor.data[idx] = config.backend.add(original_value[idx], h)

        # fn(p + h)
        fire(root)
        fn_plus = copy.deepcopy(root.data)

        # change the element back to its original value
        reference_tensor.data[idx] = original_value[idx]

        # turn off fired for operations until reference_tensor
        reset_graph(root, reference_tensor)

        # pertube one element at a time
        reference_tensor.data[idx] = config.backend.subtract(original_value[idx], h)

        # fn(p - h)
        fire(root)
        fn_minus = copy.deepcopy(root.data)

        # change the element back to its original value
        reference_tensor.data[idx] = original_value[idx]

        numerical_gradient[idx] = config.backend.divide(config.backend.subtract(fn_plus, fn_minus).sum(), config.backend.multiply(h, 2))

    # reset gradient
    reference_tensor.grad = analytical_gradient

    # dimensionality check
    if strict:
        assert analytical_gradient.shape == numerical_gradient.shape, "[gradient check] shape of analytical and numerical gradient are not equal"

    difference_norm = config.backend.linalg.norm((analytical_gradient - numerical_gradient).data)

    gradient_norm = config.backend.add(config.backend.linalg.norm(analytical_gradient.data), config.backend.linalg.norm(numerical_gradient.data))

    relative_error = difference_norm / gradient_norm

    logging.debug("[gradient check] difference error: {}, gradient norm: {}, relative error: {}\n"
                  .format(difference_norm, gradient_norm, relative_error))

    return relative_error < error_tolerance


@jaxtyped(typechecker=typechecker)
def convert_and_shuffle_dataset(
        X: Iterable,
        y: Iterable,
        shuffle: bool,
        seed: Optional[int] = None
) -> tuple[tuple, tuple]:
    if not isinstance(X, ArrayType):
        X = config.Array(X)

    if not isinstance(y, ArrayType):
        y = config.Array(y)

    dataset = list(zip(X, y))

    if shuffle:
        random.seed(seed)
        random.shuffle(dataset)

    tuple_X, tuple_y = zip(*dataset)

    return tuple_X, tuple_y


@jaxtyped(typechecker=typechecker)
def train_test_split(
        X: Iterable,
        y: Iterable,
        train_val_split: Union[tuple, float],
        shuffle: bool,
        seed: Optional[int] = None
) -> NamedTuple:
    # error checking
    if isinstance(train_val_split, (tuple, list)) and len(train_val_split) != 2:
        raise ValueError("size of train_val_split should be 2 if train_val_split is a tuple")

    if isinstance(train_val_split, (int, float)):
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

    # convert the list of Arrays to a single Array
    X_train = Tensor(config.backend_op('stack', X_train, axis=0), requires_grad=False)
    y_train = Tensor(config.backend_op('stack', y_train, axis=0), requires_grad=False)

    X_test = Tensor(config.backend_op('stack', X_test, axis=0), requires_grad=False)
    y_test = Tensor(config.backend_op('stack', y_test, axis=0), requires_grad=False)

    if do_validation_split:
        X_validation = Tensor(config.backend_op('stack', X_validation, axis=0), requires_grad=False)
        y_validation = Tensor(config.backend_op('stack', y_validation, axis=0), requires_grad=False)

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
            X: Iterable,
            y: Iterable,
            batch_size: int,
            shuffle: bool,
            seed: Optional[int] = None
    ):
        self.seed = seed

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.X, self.y = convert_and_shuffle_dataset(
            X,
            y,
            self.shuffle,
            self.seed
        )

        self.dataset_size = len(self.y) # len(self.x) can also be used
        self.mini_batch_dataset_size = self.dataset_size // self.batch_size


    def __len__(self):
        return self.mini_batch_dataset_size


    def __iter__(self):
        for batch_idx in range(0, self.dataset_size, self.batch_size):
            yield (Tensor(config.backend_op('stack', self.X[batch_idx:(batch_idx + self.batch_size)], axis=0), requires_grad=False),
                   Tensor(config.backend_op('stack', self.y[batch_idx:(batch_idx + self.batch_size)], axis=0), requires_grad=False))

