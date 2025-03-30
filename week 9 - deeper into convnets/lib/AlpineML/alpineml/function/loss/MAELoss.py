from alpineml.function.Function import Function
import mlx.core as mx


class MAELoss(Function):
    @staticmethod
    def apply(y_pred, y_true):
        return mx.abs(y_pred - y_true)

    @staticmethod
    def derivative(y_pred, y_true):
        return mx.where(y_pred > y_true, 1, -1)


mae_loss = MAELoss()
