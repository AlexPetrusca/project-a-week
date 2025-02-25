from alpineml.function.Function import Function
import mlx.core as mx

from alpineml.function.activation import Softmax


class CrossEntropyLoss(Function):
    def apply(self, y_pred, y_true, epsilon=1e-7):
        s = Softmax().apply(y_pred)
        s = mx.clip(s, epsilon, 1 - epsilon) # numerical stability
        return -y_true * mx.log(s)

    def apply_derivative(self, y_pred, y_true, epsilon=1e-7):
        s = Softmax().apply(y_pred)
        # s = mx.clip(s, epsilon, 1 - epsilon) # numerical stability
        return s - y_true