from alpineml.function.Function import Function
import mlx.core as mx


class Tanh(Function):
    def apply(self, z):
        return mx.tanh(z)

    def apply_derivative(self, z):
        return 1 - mx.tanh(z)**2
