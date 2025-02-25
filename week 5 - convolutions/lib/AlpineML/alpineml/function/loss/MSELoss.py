from alpineml.function.Function import Function


class MSELoss(Function):
    @staticmethod
    def apply(y_pred, y_true):
        return (y_pred - y_true)**2 / 2

    @staticmethod
    def derivative(y_pred, y_true):
        return y_pred - y_true


mse_loss = MSELoss()
