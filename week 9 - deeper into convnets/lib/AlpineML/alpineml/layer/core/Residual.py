from typing import Iterable

import mlx.core as mx
from alpineml.layer.Layer import Layer


class Residual(Layer):
    def __init__(self, layer: Layer, residual_dim: Iterable[int]):
        super().__init__()
        self.layer = layer
        self.input_shape = residual_dim
        self.output_shape = residual_dim

    def _link(self):
        pass

    def _forward(self, x_in: mx.array) -> mx.array:
        return x_in + self.layer.forward(x_in)

    def _backward(self, dx_out: mx.array) -> mx.array:
        return dx_out + self.layer.backward(dx_out)

