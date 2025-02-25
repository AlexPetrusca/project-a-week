import mlx.core as mx
from alpineml.layer.Layer import Layer
from alpineml.function import Function
import alpineml.function.activation as F


class Softmax(Layer):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.fn: Function = F.Softmax()
        self.temperature: float = temperature

    def _link(self):
        self.output_shape = self.input_shape

    def _forward(self, x_in: mx.array) -> mx.array:
        return self.fn(x_in, self.temperature)

    def _backward(self, dx_out: mx.array) -> mx.array:
        return self.ctx.x_out * (dx_out - mx.sum(self.ctx.x_out * dx_out, axis=1, keepdims=True))