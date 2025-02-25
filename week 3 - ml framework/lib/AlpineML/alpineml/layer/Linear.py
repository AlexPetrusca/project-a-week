import math
import mlx.core as mx
from alpineml.layer.Layer import Layer


class Linear(Layer):
    def __init__(self, input_dims: int, output_dims: int):
        super().__init__()
        # Same initialization that PyTorch uses
        scale = math.sqrt(1.0 / input_dims)
        self.W = mx.random.uniform(-scale, scale, (output_dims, input_dims))
        self.b = mx.random.uniform(-scale, scale, shape=(output_dims, 1))

    def _forward(self, x_in: mx.array) -> mx.array:
        return self.W @ x_in + self.b

    def _backward(self, dx_out: mx.array) -> mx.array:
        return self.W.transpose() @ dx_out

    def _update(self, o) -> None:
        batch_size = self.ctx.dx_out.shape[-1]
        eta = o.eta / batch_size

        dW = self.ctx.dx_out @ self.ctx.x_in.T
        db = mx.sum(self.ctx.dx_out, axis=1, keepdims=True)

        self.ctx['vW'] = o.momentum * self.ctx.get('vW', 0) - o.weight_decay * eta * self.W - eta * dW
        self.ctx['vb'] = o.momentum * self.ctx.get('vb', 0) - o.weight_decay * eta * self.b - eta * db

        self.W += self.ctx['vW']
        self.b += self.ctx['vb']
