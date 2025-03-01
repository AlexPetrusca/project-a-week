import math
from mlx import core as mx
from alpineml.layer import Layer


class MyConv2d(Layer):
    def __init__(self, out_channels: int, kernel_size: tuple | int, stride: tuple | int = 1, padding: tuple | int = 0, dilation: tuple | int = 1):
        super().__init__()
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        if isinstance(stride, int):
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

        self.in_channels = self.input_shape[2]

        h_in, w_in = self.input_shape[:2]
        w_out = math.floor((w_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        h_out = math.floor((h_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        self.output_shape = (h_out, w_out, self.out_channels)

        # size of each filter  =  in_channels x kernel_height x kernel_width
        # number of filters    =  out_channels
        # convolution of the image and weights produces the output
        # bias is applied to each output pixel
        scale = math.sqrt(1 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.params["W"] = mx.random.uniform(-scale, scale, shape=(self.out_channels, *self.kernel_size, self.in_channels))
        self.params["b"] = mx.random.uniform(-scale, scale, shape=self.output_shape)

    def _forward(self, x_in: mx.array) -> mx.array:
        x_out = mx.conv2d(x_in, self.params['W'], stride=self.stride, padding=self.padding, dilation=self.dilation)
        x_out += self.params["b"]
        return x_out

    def _backward(self, dx_out: mx.array) -> mx.array:
        # Define a function that applies convolution and returns a scalar
        def conv2d_fn(input_data, weight, stride=self.stride, padding=self.padding, dilation=self.dilation):
            return mx.sum(mx.conv2d(input_data, weight, stride, padding, dilation) * dx_out)

        # Compute gradient of convolution output w.r.t. the weights
        grad_dx_in = mx.grad(conv2d_fn, argnums=0)
        grad_dW = mx.grad(conv2d_fn, argnums=1)
        self.params["dW"] += grad_dW(self.ctx.x_in, self.params['W'])
        self.params["db"] += mx.sum(dx_out, axis=0)
        return grad_dx_in(self.ctx.x_in, self.params['W'])
