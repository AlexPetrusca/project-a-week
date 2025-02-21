from alpineml.function.Function import Function
import mlx.core as mx

class BinaryCrossEntropyLoss(Function):
    def apply(self, y_pred, y_true):
        y_pred = mx.clip(y_pred, 1e-7, 1 - 1e-7) # computational stability
        return -(y_true * mx.log(y_pred) + (1 - y_true) * mx.log(1 - y_pred))

    def apply_derivative(self, y_pred, y_true):
        y_pred = mx.clip(y_pred, 1e-7, 1 - 1e-7) # computational stability
        return (y_pred - y_true) / (y_pred * (1 - y_pred))