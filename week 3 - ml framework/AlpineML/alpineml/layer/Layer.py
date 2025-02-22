from abc import ABC, abstractmethod
from typing import Optional
import mlx.core as mx


class Layer(ABC):
    def __init__(self):
        self.ctx: LayerContext = LayerContext()

    def forward(self, x_in: mx.array, save_ctx=True) -> mx.array:
        x_out = self._forward(x_in)
        if save_ctx:
            self.ctx.x_in = x_in
            self.ctx.x_out = x_out
        return x_out

    def backward(self, dx_out: mx.array, save_ctx=True) -> mx.array:
        dx_in = self._backward(dx_out)
        if save_ctx:
            self.ctx.dx_out = dx_out
            self.ctx.dx_in = dx_in
        return dx_in

    def update(self, optimizer) -> None:
        self._update(optimizer)
        self.ctx.reset()

    @abstractmethod
    def _forward(self, x_in: mx.array) -> mx.array:
        pass

    @abstractmethod
    def _backward(self, dx_out: mx.array) -> mx.array:
        pass

    @abstractmethod
    def _update(self, optimizer) -> None:
        pass


class LayerContext(dict):
    def __init__(self):
        super().__init__()
        self.x_in: Optional[mx.array] = None
        self.x_out: Optional[mx.array] = None
        self.dx_out: Optional[mx.array] = None
        self.dx_in: Optional[mx.array] = None

    def reset(self):
        self.x_in = None
        self.x_out = None
        self.dx_out = None
        self.dx_in = None