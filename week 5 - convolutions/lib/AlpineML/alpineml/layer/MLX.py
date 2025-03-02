from mlx import core as mx
from alpineml.layer import Layer


class MLX(Layer):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def _link(self) -> None:
        x_in = mx.zeros((1, *self.input_shape))
        x_out = self.layer(x_in)
        self.output_shape = x_out.shape[1:]

    def _forward(self, x_in: mx.array) -> mx.array:
        return self.layer(x_in)

    def _backward(self, dx_out: mx.array) -> mx.array:
        def loss(x_in: mx.array):
            return mx.sum(self._forward(x_in) * dx_out)
        return mx.grad(loss, argnums=0)(self.ctx.x_in)