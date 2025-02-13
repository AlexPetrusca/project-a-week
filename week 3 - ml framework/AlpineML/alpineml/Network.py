from abc import ABC
import mlx.core as mx
from alpineml.layer.Layer import Layer


class Network(ABC):
    def __init__(self, layers=None):
        self.layers: list[Layer] = layers if layers is not None else []

    def forward(self, x: mx.array, save_ctx=False) -> mx.array:
        for layer in self.layers:
            x = layer.forward(x, save_ctx=save_ctx)
        return x

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)

    def insert_layer(self, index: int, layer: Layer) -> None:
        self.layers.insert(index, layer)
