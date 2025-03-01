from mlx import core as mx
from alpineml.layer import Layer


class Transpose(Layer):
    def __init__(self, axes: tuple):
        super().__init__()

        def invert_transpose_axes(axes):
            inv_axes = [0] * len(axes)
            for i, pos in enumerate(axes):
                inv_axes[pos] = i
            return tuple(inv_axes)

        self.axes = axes
        self.inv_axes = invert_transpose_axes(axes)

        self.batch_axes = (0, *map(lambda axis: axis + 1, self.axes))
        self.batch_inv_axes = (0, *map(lambda axis: axis + 1, self.inv_axes))

    def _link(self) -> None:
        if len(self.input_shape) != len(self.axes):
            raise ValueError("Input and output shapes must have same length.")

        self.output_shape = tuple(map(lambda axis: self.input_shape[axis], self.axes))

    def _forward(self, x_in: mx.array) -> mx.array:
        return x_in.transpose(self.batch_axes)

    def _backward(self, dx_out: mx.array) -> mx.array:
        return dx_out.transpose(self.batch_inv_axes)
