import torch as tc
import numpy as np
from mlx import core as mx
from alpineml.layer import Layer

class PyTorch(Layer):
    mps_device = tc.device("mps")

    def __init__(self, layer: tc.nn.Module):
        super().__init__()
        self.layer: tc.nn.Module = layer.to(PyTorch.mps_device)

    def _link(self) -> None:
        x_in = tc.zeros((1, *self.input_shape), device=PyTorch.mps_device)
        x_out = self.layer(x_in)
        self.output_shape = tuple(x_out.shape[1:])

        for name, param in self.layer.named_parameters(recurse=False):
            self.params[name] = param

    def _forward(self, x_in: mx.array) -> mx.array:
        t_in = PyTorch.mlx2torch(x_in)

        for name, param in self.params.items():
            self.layer._parameters[name] = tc.nn.Parameter(param.value)

        t_out = self.layer(t_in)
        self.ctx["t_out"] = t_out
        self.ctx["t_in"] = t_in

        return PyTorch.torch2mlx(t_out)

    def _backward(self, dx_out: mx.array) -> mx.array:
        tx_out = PyTorch.mlx2torch(dx_out)

        self.layer.zero_grad()
        loss = tc.sum(self.ctx["t_out"] * tx_out)
        loss.backward()
        for name, param in self.layer.named_parameters(recurse=False):
            self.params[f"d{name}"] = param.grad

        dt_in = self.ctx["t_in"].grad
        return PyTorch.torch2mlx(dt_in)

    @staticmethod
    def mlx2torch(x: mx.array) -> tc.Tensor:
        return tc.tensor(np.array(x), device=PyTorch.mps_device, requires_grad=True)

    @staticmethod
    def torch2mlx(x: tc.Tensor) -> mx.array:
        return mx.array(x.cpu().detach().numpy())