from alpineml.function.Function import Function
from alpineml.function.activation import softmax
import mlx.core as mx


class CrossEntropyLoss(Function):
    @staticmethod
    def apply(y_pred, y_true, epsilon=1e-7):
        s = softmax(y_pred)
        s = mx.clip(s, epsilon, 1 - epsilon) # numerical stability
        return -y_true * mx.log(s)

    @staticmethod
    def derivative(y_pred, y_true, epsilon=1e-7):
        s = softmax(y_pred)
        # s = mx.clip(s, epsilon, 1 - epsilon) # numerical stability
        return s - y_true


cross_entropy_loss = CrossEntropyLoss()
