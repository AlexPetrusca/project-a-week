import math
import mlx.core as mx
import mlx.nn as nn
from alpineml.layer.Layer import Layer


class MyLinear(Layer):
    def __init__(self, output_dims: int):
        super().__init__()
        self.output_shape = (output_dims,)

    def _link(self):
        self.layer = nn.Linear(self.input_shape[0], self.output_shape[0], bias=True)

    def _forward(self, x_in: mx.array) -> mx.array:
        return self.layer(x_in)

    def _backward(self, dx_out: mx.array) -> mx.array:
        def loss_fn(x, layer):
            return mx.sum(layer(x) * dx_out)

        # Compute the gradient of loss with respect to inputs
        grad_fn = mx.grad(loss_fn, argnums=0)  # For input gradient
        dx = grad_fn(self.ctx.x_in, self.layer)

        # Compute the gradient of loss with respect to layer parameters
        grad_fn_params = mx.grad(loss_fn, argnums=1)  # For layer parameter gradients
        dlayer = grad_fn_params(self.ctx.x_in, self.layer)

        learning_rate = 0.1 / dx_out.shape[0]
        for param_name, param_grad in dlayer.items():
            self.layer[param_name] = self.layer[param_name] - learning_rate * param_grad

        return dx