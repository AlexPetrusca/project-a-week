from alpineml.function.Function import Function
import mlx.core as mx


class Softmax(Function):
    def apply(self, z):
        exp_x = mx.exp(z - mx.max(z, axis=0, keepdims=True))
        return exp_x / mx.sum(exp_x, axis=0, keepdims=True)

    def apply_derivative(self, z):
        # # Partial Jacobian: only computes the diagonal elements of the Jacobian matrix.
        # #
        # # This simplified derivative is valid only when you are computing the gradient of the loss with
        # # respect to the input x and the loss function is cross-entropy loss. In this case, the off-diagonal
        # # terms of the Jacobian matrix cancel out, and you only need the diagonal terms
        # z = self.apply(z)
        # return z * (1 - z)

        # Full Jacobian:
        num_classes, batch_size = z.shape
        # Create a diagonal matrix for the diagonal terms: s_i * (1 - s_i)
        diag_terms = z * (1 - z)  # Shape: (num_classes, batch_size)
        # Create off-diagonal terms using outer product: -s_i * s_j
        off_diag_terms = -mx.einsum('ib,jb->ijb', z, z)  # Shape: (num_classes, num_classes, batch_size)
        # Combine diagonal and off-diagonal terms
        jacobian = off_diag_terms  # Start with off-diagonal terms
        # Add diagonal terms to the diagonal positions
        jacobian[mx.arange(num_classes), mx.arange(num_classes), :] += diag_terms
        return jacobian
