from alpineml.function.Function import Function


class HuberLoss(Function):
    def apply(self, y_pred, y_true):
        raise NotImplementedError()

    def apply_derivative(self, y_pred, y_true):
        raise NotImplementedError()