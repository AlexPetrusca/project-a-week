from alpineml.function.Function import Function
import mlx.core as mx


# delta - Threshold for transitioning between squared and absolute loss.
class HuberLoss(Function):
    @staticmethod
    def apply(y_pred, y_true, delta=1.0):
        residual = y_true - y_pred
        abs_residual = mx.abs(residual)
        quadratic = 0.5 * mx.square(residual)
        linear = delta * (abs_residual - 0.5 * delta)
        return mx.where(abs_residual <= delta, quadratic, linear)

    @staticmethod
    def derivative(y_pred, y_true, delta=1.0):
        residual = y_pred - y_true
        abs_residual = mx.abs(residual)
        return mx.where(abs_residual <= delta, residual, delta * mx.sign(residual))


huber_loss = HuberLoss()
