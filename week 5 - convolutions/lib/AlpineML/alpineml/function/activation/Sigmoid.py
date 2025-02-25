from alpineml.function.Function import Function
import mlx.core as mx


class Sigmoid(Function):
    @staticmethod
    def apply(z):
        return mx.sigmoid(z)

    @staticmethod
    def derivative(z):
        y = mx.sigmoid(z)
        return y * (1 - y)


sigmoid = Sigmoid()
