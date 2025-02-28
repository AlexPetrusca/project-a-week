import numpy as np
from mlx.data.datasets import load_mnist, load_cifar10, load_fashion_mnist


def one_hot_encode(y):
    encoded = np.zeros(10)
    encoded[y] = 1
    return encoded


def get_cifar10(root=None):
    tr = load_cifar10(root=root)

    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    def normalize(x):
        x = x.astype("float32") / 255.0
        return (x - mean) / std

    tr_iter = (
        tr.shuffle()
        .to_stream()
        .image_random_h_flip("image", prob=0.5)
        .pad("image", 0, 4, 4, 0.0)
        .pad("image", 1, 4, 4, 0.0)
        .image_random_crop("image", 32, 32)
        .key_transform("image", normalize)
        .prefetch(4, 4)
    )

    te = load_cifar10(root=root, train=False)
    te_iter = (
        te.to_stream()
        .key_transform("image", normalize)
    )

    return tr_iter, te_iter

def get_mnist(root=None):
    tr = load_mnist(root=root, train=True)

    def normalize(x):
        return x.astype("float32") / 255.0

    tr_iter = (
        tr.shuffle()
        .to_stream()
        .pad("image", 0, 4, 4, 0.0)
        .pad("image", 1, 4, 4, 0.0)
        .image_random_crop("image", 32, 32)
        .key_transform("image", normalize)
        .key_transform("label", one_hot_encode)
        .prefetch(4, 4)
    )

    te = load_mnist(root=root, train=False)
    te_iter = (te.to_stream()
        .key_transform("image", normalize)
        .key_transform("label", one_hot_encode)
        .pad("image", 0, 2, 2, 0.0)  # added
        .pad("image", 1, 2, 2, 0.0)  # added
    )

    return tr_iter, te_iter

def get_fashion_mnist(root=None):
    tr = load_fashion_mnist(root=root, train=True)

    def normalize(x):
        return x.astype("float32") / 255.0

    tr_iter = (
        tr.shuffle()
        .to_stream()
        .pad("image", 0, 4, 4, 0.0)
        .pad("image", 1, 4, 4, 0.0)
        .image_random_crop("image", 32, 32)
        .key_transform("image", normalize)
        .key_transform("label", one_hot_encode)
        .prefetch(4, 4)
    )

    te = load_fashion_mnist(root=root, train=False)
    te_iter = (te.to_stream()
        .key_transform("image", normalize)
        .key_transform("label", one_hot_encode)
        .pad("image", 0, 2, 2, 0.0)  # added
        .pad("image", 1, 2, 2, 0.0)  # added
    )

    return tr_iter, te_iter