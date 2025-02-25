import math
import mlx.core as mx
from alpineml.layer.Layer import Layer


class Linear(Layer):
    def __init__(self, input_dims: int, output_dims: int):
        super().__init__()
        # Same initialization that PyTorch uses
        scale = math.sqrt(1.0 / input_dims)
        self.params["W"] = mx.random.uniform(-scale, scale, (input_dims, output_dims))
        self.params["b"] = mx.random.uniform(-scale, scale, shape=(1, output_dims))

    def _forward(self, x_in: mx.array) -> mx.array:
        return x_in @ self.params["W"] + self.params["b"]

    def _backward(self, dx_out: mx.array) -> mx.array:
        self.params["dW"] += self.ctx.x_in.T @ dx_out
        self.params["db"] += mx.sum(dx_out, axis=0, keepdims=True)
        return dx_out @ self.params["W"].T
