import math
import mlx.core as mx
from alpineml.layer.Layer import Layer


# todo: add Linear function?
class Linear(Layer):
    def __init__(self, output_dims: int):
        super().__init__()
        self.output_shape = (output_dims,)

    def _link(self):
        # Same initialization that PyTorch uses
        scale = math.sqrt(1.0 / self.input_shape[0])
        self.params["W"] = mx.random.uniform(-scale, scale, (self.input_shape[0], self.output_shape[0]))
        self.params["b"] = mx.random.uniform(-scale, scale, shape=(1, self.output_shape[0]))

    def _forward(self, x_in: mx.array) -> mx.array:
        return x_in @ self.params["W"] + self.params["b"]

    def _backward(self, dx_out: mx.array) -> mx.array:
        self.params["dW"] += self.ctx.x_in.T @ dx_out
        self.params["db"] += mx.sum(dx_out, axis=0, keepdims=True)
        return dx_out @ self.params["W"].T
