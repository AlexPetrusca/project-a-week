from alpineml.function.Function import Function
import mlx.core as mx


class Silu(Function):
    @staticmethod
    def apply(z):
        return z * mx.sigmoid(z)

    @staticmethod
    def derivative(z):
        y = mx.sigmoid(z)
        return y + z * y * (1 - y)


silu = Silu()
