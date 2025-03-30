from alpineml.function.Function import Function
import mlx.core as mx


class Tanh(Function):
    @staticmethod
    def apply(z):
        return mx.tanh(z)

    @staticmethod
    def derivative(z):
        return 1 - mx.tanh(z)**2


tanh = Tanh()
