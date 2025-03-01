import math
from mlx import core as mx
from mlx import nn as nn
from alpineml.layer import Layer


class MyMaxPool2d(Layer):
    def __init__(self, kernel_size: tuple | int, stride: tuple | int = None, padding: tuple | int = 0, dilation: tuple | int = 0):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

    def _link(self) -> None:
        if len(self.input_shape) != 3:
            raise ValueError("Input shape must be 3 dimensional (H, W, C).")

        h_in, w_in = self.input_shape[:2]
        w_out = math.floor((w_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 2) / self.stride[1] + 1)
        h_out = math.floor((h_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 2) / self.stride[0] + 1)
        self.output_shape = (h_out, w_out, self.input_shape[2])

        self.pool = nn.MaxPool2d(self.kernel_size, self.stride, self.padding)

    def _forward(self, x_in: mx.array) -> mx.array:
        return self.pool(x_in)

    def _backward(self, dx_out: mx.array) -> mx.array:
        # Define a function that applies convolution and returns a scalar
        def maxpool2d_fn(input_data):
            return mx.sum(self.pool(input_data) * dx_out)

        grad_dx_in = mx.grad(maxpool2d_fn, argnums=0)
        return grad_dx_in(self.ctx.x_in)
