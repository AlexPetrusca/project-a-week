import os
import pickle
import gzip
from urllib import request

import numpy as np
import pandas as pd
from src.network import Network

def mnist(
    save_dir="/tmp",
    base_url="https://raw.githubusercontent.com/fgnt/mnist/master/",
    filename="mnist.pkl",
):
    def download_and_save(save_file):
        filename = [
            ["training_images", "train-images-idx3-ubyte.gz"],
            ["test_images", "t10k-images-idx3-ubyte.gz"],
            ["training_labels", "train-labels-idx1-ubyte.gz"],
            ["test_labels", "t10k-labels-idx1-ubyte.gz"],
        ]

        mnist = {}
        for name in filename:
            out_file = os.path.join("/tmp", name[1])
            request.urlretrieve(base_url + name[1], out_file)
        for name in filename[:2]:
            out_file = os.path.join("/tmp", name[1])
            with gzip.open(out_file, "rb") as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(
                    -1, 28 * 28
                )
        for name in filename[-2:]:
            out_file = os.path.join("/tmp", name[1])
            with gzip.open(out_file, "rb") as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open(save_file, "wb") as f:
            pickle.dump(mnist, f)

    save_file = os.path.join(save_dir, filename)
    if not os.path.exists(save_file):
        download_and_save(save_file)
    with open(save_file, "rb") as f:
        mnist = pickle.load(f)

    def preproc(x):
        return x.astype(np.float32) / 255.0

    def one_hot_encode(arr, max_value):
        arr = arr.astype(np.uint32)

        def _one_hot_encode(value):
            encoded = np.zeros(max_value + 1, dtype=np.uint32)
            encoded[value] = 1
            return encoded

        return np.array([_one_hot_encode(x) for x in arr])

    mnist["training_images"] = preproc(mnist["training_images"])
    mnist["training_labels"] = one_hot_encode(mnist["training_labels"], 9)
    mnist["test_images"] = preproc(mnist["test_images"])
    mnist["test_labels"] = one_hot_encode(mnist["test_labels"], 9)
    return (
        mnist["training_images"],
        mnist["training_labels"],
        mnist["test_images"],
        mnist["test_labels"],
    )

if __name__ == "__main__":
    train_x, train_y, test_x, test_y = mnist()  # 97% max accuracy

    train_set = []
    for x, y in zip(train_x, train_y):
        train_set.append({
            'data': x.reshape(-1, 1),
            'label': y.reshape(-1, 1)
        })

    test_set = []
    for x, y in zip(test_x, test_y):
        test_set.append({
            'data': x.reshape(-1, 1),
            'label': y.reshape(-1, 1)
        })

    print("MNIST Dataset Loaded!")

    network = Network(784, 10, [320, 160, 80, 40, 20], activation=Network.Activation.SIGMOID)
    network.train(train_set, batch_size=64, epochs=100, eta=7, validation_set=test_set)

    print("Network Trained!")