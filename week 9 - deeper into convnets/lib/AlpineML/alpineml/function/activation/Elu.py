from alpineml.function.Function import Function
import mlx.core as mx


class Elu(Function):
    @staticmethod
    def apply(z, alpha=1):
        return mx.where(z > 0, z, alpha * (mx.exp(z) - 1))

    @staticmethod
    def derivative(z, alpha=1):
        return mx.where(z > 0, 1, alpha * mx.exp(z))


elu = Elu()
