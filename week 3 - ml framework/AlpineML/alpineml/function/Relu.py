from alpineml.function.Function import Function
import mlx.core as mx


class Relu(Function):
    def apply(self, z):
        return mx.maximum(0, z)

    def apply_derivative(self, z):
        return mx.where(z > 0, 1, 0)
