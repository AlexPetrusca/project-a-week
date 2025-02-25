from alpineml.function.Function import Function
import mlx.core as mx


class Relu(Function):
    @staticmethod
    def apply(z):
        return mx.maximum(0, z)

    @staticmethod
    def derivative(z):
        return mx.where(z > 0, 1, 0)


relu = Relu()
