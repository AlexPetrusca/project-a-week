from alpineml.function.Function import Function
import mlx.core as mx


class LeakyRelu(Function):
    @staticmethod
    def apply(z, alpha=0.01):
        return mx.maximum(alpha * z, z)

    @staticmethod
    def derivative(z, alpha=0.01):
        return mx.where(z > 0, 1, alpha)


leaky_relu = LeakyRelu()
