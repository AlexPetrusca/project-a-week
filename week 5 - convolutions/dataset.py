import numpy as np
from mlx.data.datasets import load_mnist, load_cifar10, load_fashion_mnist


class StaticBuffer:
    def __init__(self, data_iter):
        data_buffer = data_iter.to_buffer()
        self.data_buffer = data_buffer.batch(len(data_buffer))[0]
        self.batch_size = 0

    def batch(self, batch_size):
        self.batch_size = batch_size
        images = self.data_buffer['image']
        labels = self.data_buffer['label']
        batched_images = np.split(images, len(images) / batch_size)
        batched_labels = np.split(labels, len(labels) / batch_size)
        self.data_buffer = []
        for image, label in zip(batched_images, batched_labels):
            self.data_buffer.append({'image': image, 'label': label})
        return self

    def reset(self):
        return self  # do nothing

    def __iter__(self):
        if self.batch_size > 0:
            return self.data_buffer.__iter__()
        else:
            return [self.data_buffer].__iter__()



def one_hot_encode(y):
    encoded = np.zeros(10)
    encoded[y] = 1
    return encoded


def get_cifar10(root=None, static=False):
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
        .key_transform("label", one_hot_encode)
        .prefetch(4, 4)
    )

    te = load_cifar10(root=root, train=False)
    te_iter = (
        te.to_stream()
        .key_transform("image", normalize)
        .key_transform("label", one_hot_encode)
    )

    if static:
        return StaticBuffer(tr_iter), StaticBuffer(te_iter)
    else:
        return tr_iter, te_iter

def get_mnist(root=None, static=False):
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

def get_fashion_mnist(root=None, static=False):
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