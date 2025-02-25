from alpineml.function.Function import Function
import mlx.core as mx


class BinaryCrossEntropyLoss(Function):
    @staticmethod
    def apply(y_pred, y_true, epsilon=1e-7):
        y_pred = mx.clip(y_pred, epsilon, 1 - epsilon) # numerical stability
        return -(y_true * mx.log(y_pred) + (1 - y_true) * mx.log(1 - y_pred))

    @staticmethod
    def derivative(y_pred, y_true, epsilon=1e-7):
        y_pred = mx.clip(y_pred, epsilon, 1 - epsilon) # numerical stability
        return (y_pred - y_true) / (y_pred * (1 - y_pred))


binary_cross_entropy_loss = BinaryCrossEntropyLoss()
