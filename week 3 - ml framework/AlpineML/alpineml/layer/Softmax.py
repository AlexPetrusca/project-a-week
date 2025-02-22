import mlx.core as mx
from alpineml.layer.Layer import Layer
from alpineml.function import Function
import alpineml.function.activation as F


class Softmax(Layer):
    def __init__(self):
        super().__init__()
        self.fn: Function = F.Softmax()

    def _forward(self, x_in: mx.array) -> mx.array:
        return self.fn(x_in)

    def _backward(self, dx_out: mx.array) -> mx.array:
        return self.ctx.x_out * (dx_out - mx.sum(self.ctx.x_out * dx_out, axis=0, keepdims=True))

    def _update(self, o) -> None:
        pass