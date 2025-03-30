import numpy as np
import mlx.data as mx_data
from mlx.data.datasets import load_mnist, load_cifar10, load_fashion_mnist


class StaticBuffer:
    def __init__(self, data):
        if isinstance(data, mx_data.Stream):
            raw_buffer = data.to_buffer()
            self.buffer = raw_buffer.batch(len(raw_buffer))[0]
            self.is_batched = False
        else:
            self.buffer = data
            self.is_batched = True

    def batch(self, batch_size):
        if self.is_batched:
            raise RuntimeError('Cannot batch a batched dataset')

        batched_images = np.split(self.buffer['image'], len(self) / batch_size)
        batched_labels = np.split(self.buffer['label'], len(self) / batch_size)

        batched_buffer = []
        for image, label in zip(batched_images, batched_labels):
            batched_buffer.append({'image': image, 'label': label})
        return StaticBuffer(batched_buffer)

    def shuffle(self):
        if self.is_batched:
            np.random.shuffle(self.buffer)
        else:
            permutation = np.random.permutation(len(self))
            self.buffer['image'] = self.buffer['image'][permutation]
            self.buffer['label'] = self.buffer['label'][permutation]
        return self

    def to_buffer(self):
        return self  # do nothing

    def reset(self):
        return self  # do nothing

    def __iter__(self):
        if self.is_batched:
            return self.buffer.__iter__()
        else:
            return [self.buffer].__iter__()

    def __len__(self):
        if self.is_batched:
            return len(self.buffer)
        else:
            return len(self.buffer['image'])

    def __getitem__(self, index):
        if self.is_batched:
            return self.buffer[index]
        else:
            return {
                'image': self.buffer['image'][index],
                'label': self.buffer['label'][index]
            }


def one_hot_encode(y):
    encoded = np.zeros(10)
    encoded[y] = 1
    return encoded


def get_cifar10(root=None, static=False):
    tr = load_cifar10(root=root)

    # mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    # std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    def normalize(x):
        x = x.astype("float32") / 255.0
        # return (x - mean) / std
        return x

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

    if static:
        return StaticBuffer(tr_iter), StaticBuffer(te_iter)
    else:
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

    if static:
        return StaticBuffer(tr_iter), StaticBuffer(te_iter)
    else:
        return tr_iter, te_iter