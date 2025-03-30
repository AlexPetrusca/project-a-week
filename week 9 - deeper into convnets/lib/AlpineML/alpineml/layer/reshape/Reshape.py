import math
from mlx import core as mx
from alpineml.layer import Layer


class Reshape(Layer):
    def __init__(self, output_shape: tuple | int):
        super().__init__()
        if isinstance(output_shape, int):
            self.output_shape = (output_shape,)
        else:
            self.output_shape = output_shape

    def _link(self) -> None:
        if math.prod(self.output_shape) != math.prod(self.input_shape):
            raise ValueError(f"Input shape {self.input_shape} can't be reshaped into output shape {self.output_shape}.")

    def _forward(self, x_in: mx.array) -> mx.array:
        return x_in.reshape(-1, *self.output_shape)

    def _backward(self, dx_out: mx.array) -> mx.array:
        return dx_out.reshape(-1, *self.input_shape)
