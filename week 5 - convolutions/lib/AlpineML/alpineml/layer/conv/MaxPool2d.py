import math
from mlx import core as mx
from alpineml.layer import Layer


class MaxPool2d(Layer):
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
            raise ValueError("Input shape must be 3 dimensional (C, H, W).")

        h_in, w_in = self.input_shape[1:]
        w_out = math.floor((w_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        h_out = math.floor((h_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        self.output_shape = (self.input_shape[0], h_out, w_out)

    # todo: implement dilation
    def _forward(self, x_in: mx.array) -> mx.array:
        x_out = mx.zeros((x_in.shape[0], *self.output_shape))
        max_indices = mx.zeros((x_in.shape[0], self.output_shape[0], self.output_shape[1] * self.output_shape[2]))

        # Pad the input
        x_in_padded = mx.pad(x_in, ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])))

        h_out, w_out = self.output_shape[1:]
        for y in range(h_out):
            for x in range(w_out):
                h_start = y * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = x * self.stride[1]
                w_end = w_start + self.kernel_size[1]

                # write max value to output
                window = x_in_padded[:, :, h_start:h_end, w_start:w_end]
                x_out[:, :, y, x] = mx.max(window, axis=(2, 3))

                # write max index to ctx.max_indices
                window_flat = window.reshape(window.shape[0], window.shape[1], -1)  # Flatten spatial dimensions
                max_idx = mx.argmax(window_flat, axis=-1)  # Get the index of the max value
                max_indices[:, :, y * w_out + x] = max_idx

        self.ctx['max_indices'] = max_indices
        return x_out

    # todo: implement padding
    # todo: implement dilation
    def _backward(self, dx_out: mx.array) -> mx.array:
        dx_in = mx.zeros(self.ctx.x_in.shape)
        max_indices = self.ctx['max_indices']
        h_out, w_out = self.output_shape[1:]
        for y in range(h_out):
            for x in range(w_out):
                h_start = y * self.stride[0]
                h_end = min(h_start + self.kernel_size[0], self.input_shape[1])
                w_start = x * self.stride[1]
                w_end = min(w_start + self.kernel_size[1], self.input_shape[2])

                # todo: mx.newaxis makes shit pretty hard to understand - ur resizing implicitly - revise
                # a[..., i, mx.newaxis, ...] basically means select i but keep its dimension
                indices = max_indices[:, :, y * w_out + x, mx.newaxis]
                flat_mask = mx.where(indices == mx.arange((h_end - h_start) * (w_end - w_start)), 1, 0)
                mask = flat_mask.reshape(-1, self.input_shape[0], h_end - h_start, w_end - w_start)
                grad = dx_out[:, :, y, mx.newaxis, x, mx.newaxis]
                dx_in[:, :, h_start:h_end, w_start:w_end] += grad * mask

        # # todo: passing None is incorrect (Bad LLM!)
        # # Remove padding from the gradient
        # if self.padding[0] > 0 or self.padding[1] > 0:
        #     dx_in = dx_in[
        #         :, :,
        #         self.padding[0]:-self.padding[0] if self.padding[0] > 0 else None,
        #         self.padding[1]:-self.padding[1] if self.padding[1] > 0 else None,
        #     ]

        return dx_in