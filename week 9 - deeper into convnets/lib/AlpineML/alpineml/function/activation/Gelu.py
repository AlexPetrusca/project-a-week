from alpineml.function.Consts import SQRT_2, SQRT_PI
from alpineml.function.Function import Function
import mlx.core as mx


class Gelu(Function):
    @staticmethod
    def apply(z, approx = "none"):
        match approx:
            case "none":
                return Gelu._exact(z)
            case "tanh":
                return Gelu._tanh(z)
            case "sigmoid":
                return Gelu._sigmoid(z)
            case _:
                raise ValueError("Invalid approximation type")

    @staticmethod
    def derivative(z, approx = "none"):
        match approx:
            case "none":
                return Gelu._exact_derivative(z)
            case "tanh":
                return Gelu._tanh_derivative(z)
            case "sigmoid":
                return Gelu._sigmoid_derivative(z)
            case _:
                raise ValueError("Invalid approximation type")

    @staticmethod
    def _exact(z):
        return 0.5 * z * (1 + mx.erf(z / SQRT_2))

    @staticmethod
    def _tanh(z):
        f = SQRT_2 / SQRT_PI * (z + 0.044715 * z ** 3)
        return 0.5 * z * (1 + mx.tanh(f))

    @staticmethod
    def _sigmoid(z):
        return z * mx.sigmoid(1.702 * z)

    @staticmethod
    def _exact_derivative(z):
        phi = mx.exp(-0.5 * z ** 2) / (SQRT_2 * SQRT_PI)  # PDF of standard Gaussian
        Phi = 0.5 * (1 + mx.erf(z / SQRT_2))  # CDF of standard Gaussian
        return Phi + z * phi

    @staticmethod
    def _tanh_derivative(z):
        f = SQRT_2 / SQRT_PI * (z + 0.044715 * z ** 3)
        fp = SQRT_2 / SQRT_PI * (1 + 1.134145 * z ** 2)
        return 0.5 * (1 + mx.tanh(f)) + 0.5 * z * (1 / mx.cosh(f) ** 2) * fp

    @staticmethod
    def _sigmoid_derivative(z):
        y = mx.sigmoid(1.702 * z)
        return y + (1.702 * z) * y * (1 - y)


gelu = Gelu()
