from abc import ABC
from typing import Optional
from alpineml.Network import Network
import mlx.core as mx
from alpineml.function.Function import Function


class Optimizer(ABC):
    def __init__(self):
        self.network: Optional[Network] = None
        self.loss_fn: Optional[Function] = None

    def train_network(self, x: mx.array, y: mx.array):
        y_pred = x
        for layer in self.network.layers:
            y_pred = layer.forward(y_pred)

        grads = []  # delete me
        grad = self.loss_fn.derivative(y_pred, y)
        for layer in reversed(self.network.layers):
            grads.append(grad)  # delete me
            grad = layer.backward(grad)

        for layer in self.network.layers:
            layer.update(2 / x.shape[1])

    def bind_network(self, network: Network) -> None:
        self.network = network

    def bind_loss_fn(self, loss_fn: Function) -> None:
        self.loss_fn = loss_fn
