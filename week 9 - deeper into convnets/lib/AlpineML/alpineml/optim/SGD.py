import mlx.core as mx
from alpineml.optim import Optimizer


class SGD(Optimizer):
    def __init__(self, eta=1.0, momentum=0.0, weight_decay=0.0):
        super().__init__()
        self.eta: float = eta
        self.momentum: float = momentum
        self.weight_decay: float = weight_decay

    def step(self, x: mx.array, y: mx.array):
        # feed forward
        y_pred = self.network.forward(x, save_ctx=True)

        # backpropagate
        grad = self.loss_fn.derivative(y_pred, y)
        for layer in reversed(self.network.layers):
            grad = layer.backward(grad)

        # update layers
        for i, layer in enumerate(self.network.layers):
            if layer.trainable:
                batch_size = layer.ctx.dx_out.shape[0]
                eta = self.eta / batch_size
                for param in layer.params:
                    v_key = f"v{param.name}_{i}"
                    self.ctx[v_key] = self.momentum * self.ctx.get(v_key, 0) - self.weight_decay * eta * param.value - eta * param.grad
                    param.value = param.value + self.ctx[v_key]
            layer.params.zero_grad()

        return grad # gradient for the input
