from alpineml.function.Function import Function
import mlx.core as mx


class MAELoss(Function):
    def apply(self, y_pred, y_true):
        return mx.abs(y_pred - y_true)

    def apply_derivative(self, y_pred, y_true):
        return mx.where(y_pred > y_true, 1, -1)