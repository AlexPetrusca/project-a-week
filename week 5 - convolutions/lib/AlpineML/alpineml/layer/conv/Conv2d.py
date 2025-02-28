import math
from mlx import core as mx
from alpineml.layer import Layer


class Conv2d(Layer):
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
            raise ValueError("Input shape must be 3 dimensional (C, H, W).")

        self.in_channels = self.input_shape[0]

        h_in, w_in = self.input_shape[1:]
        w_out = math.floor((w_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        h_out = math.floor((h_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        self.output_shape = (self.out_channels, h_out, w_out)

        # size of each filter  =  in_channels x kernel_height x kernel_width
        # number of filters    =  out_channels
        # convolution of the image and weights produces the output
        # bias is applied to each output pixel
        scale = math.sqrt(1 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.params["W"] = mx.random.uniform(-scale, scale, shape=(self.out_channels, self.in_channels, *self.kernel_size))
        self.params["b"] = mx.random.uniform(-scale, scale, shape=self.output_shape)

    # todo: implement dilation
    def _forward(self, x_in: mx.array) -> mx.array:
        # x_out = mx.zeros((x_in.shape[0], *self.output_shape))
        #
        # # Pad the input
        # x_in_padded = mx.pad(x_in, ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])))
        #
        # h_out, w_out = self.output_shape[1:]
        # for y in range(h_out):
        #     for x in range(w_out):
        #         h_start = y * self.stride[0]
        #         h_end = h_start + self.kernel_size[0]
        #         w_start = x * self.stride[1]
        #         w_end = w_start + self.kernel_size[1]
        #
        #         # convolve
        #         window = x_in_padded[:, :, h_start:h_end, w_start:w_end]
        #         for c_out in range(self.out_channels):
        #             kernel = self.params["W"][c_out]
        #             x_out[:, c_out:c_out + 1, y, x] += mx.sum(window * kernel, axis=(2, 3))
        #
        # x_out += self.params["b"]
        # return x_out

        x_in = x_in.transpose((0, 2, 3, 1))  # (N, C_in, H, W) --> (N, H, W, C_in)
        W = self.params['W'].transpose((0, 2, 3, 1)) # (C_out, C_in, H, W) --> (C_out, H, W, C_in)
        x_out = mx.conv2d(x_in, W, stride=self.stride, padding=self.padding, dilation=self.dilation)
        x_out = x_out.transpose((0, 3, 1, 2)) # (N, H, W, C_out) --> (N, C_out, H, W)
        x_out += self.params["b"]
        return x_out

    # todo: implement padding
    # todo: implement dilation
    def _backward(self, dx_out: mx.array) -> mx.array:
        # dx_in = mx.zeros(self.ctx.x_in.shape)
        # h_out, w_out = self.output_shape[1:]
        # for y in range(h_out):
        #     for x in range(w_out):
        #         h_start = y * self.stride[0]
        #         h_end = h_start + self.kernel_size[0]
        #         w_start = x * self.stride[1]
        #         w_end = w_start + self.kernel_size[1]
        #
        #         for c_out in range(self.out_channels):
        #             kernel = self.params["W"][c_out]
        #             grad = dx_out[:, c_out:c_out+1, y:y+1, x:x+1]
        #             dx_in[:, :, h_start:h_end, w_start:w_end] += grad * kernel
        #
        #             self.params["dW"] += mx.sum(self.ctx.x_in[:, :, h_start:h_end, w_start:w_end] * grad, axis=0)
        #
        # # todo: passing None is incorrect (Bad LLM!)
        # # # Remove padding from the gradient
        # # if self.padding[0] > 0 or self.padding[1] > 0:
        # #     dx_in = dx_in[
        # #         :, :,
        # #         self.padding[0]:-self.padding[0] if self.padding[0] > 0 else None,
        # #         self.padding[1]:-self.padding[1] if self.padding[1] > 0 else None,
        # #     ]
        #
        # self.params["db"] += mx.sum(dx_out, axis=0)
        # return dx_in

        # todo: doesnt work...
        # Define a function that applies convolution and returns a scalar
        def conv2d_fn(input_data, weight, stride=self.stride, padding=self.padding, dilation=self.dilation):
            # return mx.conv2d(input_data, weight, stride, padding, dilation).sum()
            return mx.conv2d(input_data, weight, stride, padding, dilation).mean()
        # Compute gradient of convolution output w.r.t. the weights
        x_in = self.ctx.x_in.transpose((0, 2, 3, 1))  # (N, C_in, H, W) --> (N, H, W, C_in)
        W = self.params['W'].transpose((0, 2, 3, 1))  # (C_out, C_in, H, W) --> (C_out, H, W, C_in)
        grad_dx_in = mx.grad(conv2d_fn, argnums=0)
        grad_dW = mx.grad(conv2d_fn, argnums=1)
        self.params["dW"] += grad_dW(x_in, W).transpose((0, 3, 1, 2))
        self.params["db"] += mx.sum(dx_out, axis=0)
        return grad_dx_in(x_in, W).transpose((0, 3, 1, 2))
