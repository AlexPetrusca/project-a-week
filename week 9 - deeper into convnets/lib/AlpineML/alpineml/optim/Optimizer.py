from abc import ABC, abstractmethod
from typing import Optional

import mlx.core as mx

from alpineml import Network
from alpineml.function import Function


class Optimizer(ABC):
    def __init__(self):
        self.network: Optional[Network] = None
        self.loss_fn: Optional[Function] = None
        self.ctx: dict = {}

    def bind_network(self, network: Network) -> None:
        self.network = network

    def bind_loss_fn(self, loss_fn: Function) -> None:
        self.loss_fn = loss_fn

    @abstractmethod
    def step(self, x: mx.array, y: mx.array):
        pass