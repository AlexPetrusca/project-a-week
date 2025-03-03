import inspect
from mlx import core as mx
from mlx import nn as nn
from alpineml.layer import Layer


class MLX(Layer):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

        def loss_fn(x_in, dx_out):
            return mx.sum(self.layer(x_in) * dx_out)

        self.value_grad_fn = mx.grad(loss_fn, argnums=0)
        self.param_grad_fn = nn.value_and_grad(self.layer, loss_fn)

    def _link(self) -> None:
        x_in = mx.zeros((1, *self.input_shape))
        x_out = self.layer(x_in)
        self.output_shape = x_out.shape[1:]

        for name, value in self.layer.trainable_parameters().items():
            self.params[name] = value

    def _forward(self, x_in: mx.array) -> mx.array:
        for name, param in self.params.items():
            self.layer[name] = param.value
        return self.layer(x_in)

    def _backward(self, dx_out: mx.array) -> mx.array:
        if len(self.params) > 0:
            _, grads = self.param_grad_fn(self.ctx.x_in, dx_out)
            for name, value in grads.items():
                self.params[f"d{name}"] = value

        return self.value_grad_fn(self.ctx.x_in, dx_out)