from alpineml.function.Function import Function
import mlx.core as mx


class Swish(Function):
    @staticmethod
    def apply(z, beta=1):
        return z * mx.sigmoid(beta * z)

    @staticmethod
    def derivative(z, beta=1):
        y = mx.sigmoid(beta * z)
        return y + (beta * z) * y * (1 - y)


swish = Swish()
