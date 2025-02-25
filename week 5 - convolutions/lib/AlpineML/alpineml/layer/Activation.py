import mlx.core as mx
from alpineml.layer.Layer import Layer
from alpineml.function.Function import Function


class Activation(Layer):
    def __init__(self, fn: Function):
        super().__init__()
        self.fn: Function = fn

    def _forward(self, x_in: mx.array) -> mx.array:
        return self.fn(x_in)

    def _backward(self, dx_out: mx.array) -> mx.array:
        return dx_out * self.fn.derivative(self.ctx.x_in)