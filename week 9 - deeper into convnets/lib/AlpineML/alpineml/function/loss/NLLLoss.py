from alpineml.function.Function import Function
import mlx.core as mx


class NLLLoss(Function):
    @staticmethod
    def apply(y_pred, y_true, epsilon=1e-7):
        y_pred = mx.clip(y_pred, epsilon, 1 - epsilon) # numerical stability
        return -y_true * mx.log(y_pred)

    @staticmethod
    def derivative(y_pred, y_true, epsilon=1e-7):
        y_pred = mx.clip(y_pred, epsilon, 1 - epsilon) # numerical stability
        return -y_true / y_pred


nll_loss = NLLLoss()
