from abc import ABC
from typing import Optional
from alpineml.Network import Network
import mlx.core as mx
from alpineml.function.Function import Function


class Optimizer(ABC):
    def __init__(self):
        self.network: Optional[Network] = None
        self.loss_fn: Optional[Function] = None
        self.eta: float = 1.0
        self.momentum: float = 0.0
        self.weight_decay: float = 0.0

    def train_network(self, x: mx.array, y: mx.array):
        # feed forward
        y_pred = self.network.forward(x, save_ctx=True)
        # backpropagate
        grad = self.loss_fn.derivative(y_pred, y)
        for layer in reversed(self.network.layers):
            grad = layer.backward(grad)
        # update layers
        for layer in self.network.layers:
            layer.update(self)

    def bind_network(self, network: Network) -> None:
        self.network = network

    def bind_loss_fn(self, loss_fn: Function) -> None:
        self.loss_fn = loss_fn

    def set_learning_rate(self, eta: float) -> None:
        self.eta = eta

    def set_weight_decay(self, weight_decay: float) -> None:
        self.weight_decay = weight_decay

    def set_momentum(self, momentum: float) -> None:
        self.momentum = momentum