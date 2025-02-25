from alpineml.function.Function import Function
import mlx.core as mx


class Gelu(Function):
    def __init__(self, approx = "none"):
        super().__init__()

        SQRT_2 = mx.sqrt(2)
        SQRT_PI = mx.sqrt(mx.pi)

        def _exact(z):
            return 0.5 * z * (1 + mx.erf(z / SQRT_2))

        def _exact_derivative(z):
            phi = mx.exp(-0.5 * z**2) / (SQRT_2 * SQRT_PI)  # PDF of standard Gaussian
            Phi = 0.5 * (1 + mx.erf(z / SQRT_2))  # CDF of standard Gaussian
            return Phi + z * phi

        def _tanh(z):
            f = SQRT_2 / SQRT_PI * (z + 0.044715 * z**3)
            return 0.5 * z * (1 + mx.tanh(f))

        def _tanh_derivative(z):
            f = SQRT_2 / SQRT_PI * (z + 0.044715 * z**3)
            fp = SQRT_2 / SQRT_PI * (1 + 1.134145 * z**2)
            return 0.5 * (1 + mx.tanh(f)) + 0.5 * z * (1 / mx.cosh(f)**2) * fp

        def _sigmoid(z):
            return z * mx.sigmoid(1.702 * z)

        def _sigmoid_derivative(z):
            y = mx.sigmoid(1.702 * z)
            return y + (1.702 * z) * y * (1 - y)

        match approx:
            case "none":
                self.apply = _exact
                self.apply_derivative = _exact_derivative
            case "tanh":
                self.apply = _tanh
                self.apply_derivative = _tanh_derivative
            case "sigmoid":
                self.apply = _sigmoid
                self.apply_derivative = _sigmoid_derivative
            case _:
                raise ValueError("Invalid approximation type")

    def apply(self, z):
        raise NotImplementedError()

    def apply_derivative(self, z):
        raise NotImplementedError()