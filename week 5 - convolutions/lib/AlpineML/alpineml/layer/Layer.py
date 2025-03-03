from abc import ABC, abstractmethod
from typing import Optional
import mlx.core as mx


class Layer(ABC):
    def __init__(self):
        self.ctx: Layer.Context = Layer.Context()
        self.params: Layer.Parameters = Layer.Parameters()
        self.input_shape: Optional[tuple] = None
        self.output_shape: Optional[tuple] = None

    def link(self, input_shape: tuple | int) -> None:
        if isinstance(input_shape, int):
            self.input_shape = (input_shape,)
        else:
            self.input_shape = input_shape
        self._link()

    def forward(self, x_in: mx.array, save_ctx=True) -> mx.array:
        x_out = self._forward(x_in)
        if save_ctx:
            self.ctx.x_in = x_in
            self.ctx.x_out = x_out
        return x_out

    def backward(self, dx_out: mx.array, save_ctx=True) -> mx.array:
        dx_in = self._backward(dx_out)
        if save_ctx:
            self.ctx.dx_out = dx_out
            self.ctx.dx_in = dx_in
        return dx_in

    @abstractmethod
    def _link(self):
        pass

    @abstractmethod
    def _forward(self, x_in: mx.array) -> mx.array:
        pass

    @abstractmethod
    def _backward(self, dx_out: mx.array) -> mx.array:
        pass

    class Context(dict):
        def __init__(self):
            super().__init__()
            self.x_in: Optional[mx.array] = None
            self.x_out: Optional[mx.array] = None
            self.dx_out: Optional[mx.array] = None
            self.dx_in: Optional[mx.array] = None

        def reset(self):
            self.x_in = None
            self.x_out = None
            self.dx_out = None
            self.dx_in = None

    class Parameter:
        def __init__(self, name: str, value: mx.array = 0, gradient: mx.array = 0):
            self.name: str = name
            self.value: mx.array = value
            self.grad: mx.array = gradient

        def zero_grad(self):
            self.grad = 0

    class Parameters:
        def __init__(self):
            super().__init__()
            self._params: dict[str, Layer.Parameter] = {}

        def __getitem__(self, key: str) -> Optional[mx.array]:
            is_grad = key.startswith("d")
            key = key[1:] if key.startswith("d") else key
            if not self._params.get(key):
                raise KeyError(f"Parameter {key} is not defined.")
            return self._params[key].grad if is_grad else self._params[key].value

        def __setitem__(self, key: str, val: mx.array) -> None:
            is_grad = key.startswith("d")
            key = key[1:] if key.startswith("d") else key
            if not self._params.get(key):
                if is_grad:
                    raise KeyError(f"Parameter {key} is not defined.")
                else:
                    self._params[key] = Layer.Parameter(key)
            if is_grad:
                self._params[key].grad = val
            else:
                self._params[key].value = val

        def __iter__(self):
            return self._params.values().__iter__()

        def __str__(self) -> str:
            return str(self._params)

        def __repr__(self) -> str:
            return repr(self._params)

        def __len__(self) -> int:
            return len(self._params)

        def items(self):
            return self._params.items()

        def zero_grad(self) -> None:
            for param in self:
                param.zero_grad()
