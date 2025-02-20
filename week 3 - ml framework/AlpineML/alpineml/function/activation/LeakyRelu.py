from alpineml.function.Function import Function
import mlx.core as mx


class LeakyRelu(Function):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def apply(self, z):
        return mx.maximum(self.alpha * z, z)

    def apply_derivative(self, z):
        return mx.where(z > 0, 1, self.alpha)
