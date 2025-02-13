from alpineml.function.Function import Function


class MeanSquareError(Function):
    def apply(self, y_pred, y_true):
        return (y_pred - y_true)**2 / 2

    def apply_derivative(self, y_pred, y_true):
        return y_pred - y_true
