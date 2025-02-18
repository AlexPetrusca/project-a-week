import math
import numpy as np
import mlx.core as mx
from alpineml.layer.Layer import Layer



class Linear(Layer):
    def __init__(self, input_dims: int, output_dims: int):
        super().__init__()
        # Same initialization that PyTorch uses
        scale = math.sqrt(1.0 / input_dims)
        self.W = mx.random.uniform(-scale, scale, (output_dims, input_dims))
        self.b = mx.random.uniform(-scale, scale, shape=(output_dims, 1))
        # # Custom initialization
        # self.W = mx.array(np.random.randn(output_dims, input_dims) * 0.01)
        # self.b = mx.array(np.zeros((output_dims, 1)))

    def _forward(self, x_in: mx.array) -> mx.array:
        return self.W @ x_in + self.b

    def _backward(self, dx_out: mx.array) -> mx.array:
        return self.W.transpose() @ dx_out

    def _update(self, scale: float) -> None:
        self.W -= scale * self.ctx.dx_out @ self.ctx.x_in.T
        self.b -= scale * mx.sum(self.ctx.dx_out, axis=1, keepdims=True)
        # self.b -= scale * self.ctx.dx_out @ mx.ones((self.ctx.x_in.shape[1], 1))
