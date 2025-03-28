from alpineml.function.Function import Function
import mlx.core as mx


class Softmax(Function):
    @staticmethod
    def apply(z, temperature=1.0):
        z -= mx.max(z, axis=1, keepdims=True) # numerical stability
        exp_z = mx.exp(z / temperature)
        return exp_z / mx.sum(exp_z, axis=1, keepdims=True)

    # todo: this isn't even needed practically - the Softmax layer doesn't need this derivative during backpropagation.
    @staticmethod
    def derivative(z, temperature=1.0):
        # Partial Jacobian: only computes the diagonal elements of the Jacobian matrix.
        #
        # This simplified derivative is valid only when you are computing the gradient of the loss with
        # respect to the input x and the loss function is cross-entropy loss. In this case, the off-diagonal
        # terms of the Jacobian matrix cancel out, and you only need the diagonal terms
        z = Softmax.apply(z, temperature)
        return z * (1 - z)


softmax = Softmax()
