from abc import ABC
import mlx.core as mx
from alpineml.layer.Layer import Layer


class Network(ABC):
    def __init__(self, input_shape, layers=None):
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.layers: list[Layer] = []
        if layers is not None:
            for layer in layers:
                self.add_layer(layer)

    def forward(self, x: mx.array, save_ctx=False) -> mx.array:
        for layer in self.layers:
            x = layer.forward(x, save_ctx=save_ctx)
        return x

    def add_layer(self, layer: Layer) -> None:
        layer.link(self.output_shape)
        self.layers.append(layer)
        self.output_shape = layer.output_shape
