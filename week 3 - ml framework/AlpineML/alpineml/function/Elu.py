from alpineml.function.Function import Function
import mlx.core as mx


class Elu(Function):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha

    def apply(self, z):
        return mx.where(z > 0, z, self.alpha * (mx.exp(z) - 1))

    def apply_derivative(self, z):
        return mx.where(z > 0, 1, self.alpha * mx.exp(z))
