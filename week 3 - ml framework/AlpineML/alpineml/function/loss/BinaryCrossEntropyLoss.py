from alpineml.function.Function import Function
import mlx.core as mx

# todo: even with epsilon=1e-7, this can output massive losses / gradients
class BinaryCrossEntropyLoss(Function):
    def apply(self, y_pred, y_true, epsilon=1e-7):
        y_pred = mx.clip(y_pred, epsilon, 1 - epsilon) # numerical stability
        return -(y_true * mx.log(y_pred) + (1 - y_true) * mx.log(1 - y_pred))

    def apply_derivative(self, y_pred, y_true, epsilon=1e-7):
        y_pred = mx.clip(y_pred, epsilon, 1 - epsilon) # numerical stability
        return (y_pred - y_true) / (y_pred * (1 - y_pred))