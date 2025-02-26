import math
from mlx import core as mx
from alpineml.layer import Layer


class Flatten(Layer):
    def _link(self) -> None:
        self.output_shape = (math.prod(self.input_shape),)

    def _forward(self, x_in: mx.array) -> mx.array:
        return x_in.reshape(-1, *self.output_shape)

    def _backward(self, dx_out: mx.array) -> mx.array:
        return dx_out.reshape(-1, *self.input_shape)
