import math


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._children = list(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        if self.label:
            return f"Value(label={self.label}, data={self.data:.4f})"
        else:
            return f"Value(data={self.data:.4f})"

    def tanh(self):
        t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        # t = math.tanh(x)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += out.grad * (1 - out.data**2)
        out._backward = _backward

        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward

        return out

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward

        return out

    def __radd__(self, other):
        return Value(other) + self

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * (-1.0)
        out._backward = _backward

        return out

    def __rsub__(self, other):
        return Value(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return Value(other) * self

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), '/')

        def _backward():
            self.grad += out.grad * (1 / other.data)
            other.grad += out.grad * (-self.data / other.data**2)
        out._backward = _backward

        return out

    def __rtruediv__(self, other):
        return Value(other) / self

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data**other.data, (self, other), '**')

        def _backward():
            self.grad += out.grad * (other.data / (self.data + 0.01) * out.data)
            # other.grad += out.grad * (math.log(self.data) * out.data)
        out._backward = _backward

        return out

    def __rpow__(self, other):
        return Value(other) ** self

    def __neg__(self):
        out = Value(-self.data, (self,), '-')

        def _backward():
            self.grad += out.grad * (-1.0)
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(root):
            if root not in visited:
                visited.add(root)
                for child in root._children:
                    build_topo(child)
                topo.append(root)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        visited = set()
        def _zero_grad(root):
            if root not in visited:
                visited.add(root)
                root.grad = 0.0
                for child in root._children:
                    _zero_grad(child)
        _zero_grad(self)