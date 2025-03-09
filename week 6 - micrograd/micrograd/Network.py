import random
from micrograd.Value import Value


class MLP:
    def __init__(self, indim, outdims):
        sz = [indim] + outdims
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(outdims))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class Layer:
    def __init__(self, indim, outdim):
        self.neurons = [Neuron(indim) for _ in range(outdim)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class Neuron:
    def __init__(self, indim):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(indim)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum([wi * xi for wi, xi in zip(self.w, x)], self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]