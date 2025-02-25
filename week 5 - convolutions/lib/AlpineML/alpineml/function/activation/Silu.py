from alpineml.function.Function import Function
import mlx.core as mx


class Silu(Function):
    def apply(self, z):
        return z * mx.sigmoid(z)

    def apply_derivative(self, z):
        y = mx.sigmoid(z)
        return y + z * y * (1 - y)
