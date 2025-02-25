import math
import mlx.core as mx
from alpineml.layer.Layer import Layer


class Linear(Layer):
    def __init__(self, input_dims: int, output_dims: int):
        super().__init__()
        # Same initialization that PyTorch uses
        scale = math.sqrt(1.0 / input_dims)
        self.W = mx.random.uniform(-scale, scale, (input_dims, output_dims))
        self.b = mx.random.uniform(-scale, scale, shape=(1, output_dims))

    def _forward(self, x_in: mx.array) -> mx.array:
        return x_in @ self.W + self.b

    def _backward(self, dx_out: mx.array) -> mx.array:
        return dx_out @ self.W.T

    def _update(self, o) -> None:
        batch_size = self.ctx.dx_out.shape[0]
        eta = o.eta / batch_size

        dW = self.ctx.x_in.T @ self.ctx.dx_out
        db = mx.sum(self.ctx.dx_out, axis=0, keepdims=True)

        self.ctx['vW'] = o.momentum * self.ctx.get('vW', 0) - o.weight_decay * eta * self.W - eta * dW
        self.ctx['vb'] = o.momentum * self.ctx.get('vb', 0) - o.weight_decay * eta * self.b - eta * db

        self.W += self.ctx['vW']
        self.b += self.ctx['vb']
