from alpineml.function.Function import Function
import mlx.core as mx


class Sigmoid(Function):
    def apply(self, z):
        return mx.sigmoid(z)

    def apply_derivative(self, z):
        y = mx.sigmoid(z)
        return y * (1 - y)
