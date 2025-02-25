from alpineml.function.Function import Function
import mlx.core as mx


class Swish(Function):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def apply(self, z):
        return z * mx.sigmoid(self.beta * z)

    def apply_derivative(self, z):
        y = mx.sigmoid(self.beta * z)
        return y + (self.beta * z) * y * (1 - y)
