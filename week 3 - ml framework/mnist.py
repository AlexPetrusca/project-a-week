import gzip
import os
import pickle
from urllib import request

import numpy as np
import mlx.core as mx

from alpineml import Network, Optimizer
from alpineml.function.loss import MSELoss
from alpineml.function.activation import LeakyRelu, Sigmoid
from alpineml.layer import Linear, Activation


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
            encoded = np.zeros(max_value + 1)
            encoded[value] = 1
            return encoded

        return np.array([_one_hot_encode(x) for x in arr])

    mnist["training_images"] = preproc(mnist["training_images"])
    mnist["training_labels"] = one_hot_encode(mnist["training_labels"], 9)
    mnist["test_images"] = preproc(mnist["test_images"])
    mnist["test_labels"] = one_hot_encode(mnist["test_labels"], 9)
    return (
        mnist["training_images"].T,
        mnist["training_labels"].T,
        mnist["test_images"].T,
        mnist["test_labels"].T,
    )


def fashion_mnist(save_dir="/tmp"):
    return mnist(
        save_dir,
        base_url="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        filename="fashion_mnist.pkl",
    )

train_x, train_y, test_x, test_y = map(mx.array, mnist()) # 97% max accuracy
# train_x, train_y, test_x, test_y = map(mx.array, fashion_mnist()) # 87% max accuracy

network = Network()
network.add_layer(Linear(784, 320))
network.add_layer(Activation(LeakyRelu()))
network.add_layer(Linear(320, 160))
network.add_layer(Activation(LeakyRelu()))
network.add_layer(Linear(160, 80))
network.add_layer(Activation(LeakyRelu()))
network.add_layer(Linear(80, 10))
network.add_layer(Activation(Sigmoid()))

optimizer = Optimizer()
optimizer.bind_network(network)
optimizer.bind_loss_fn(MSELoss())
optimizer.bind_learning_rate(1)

def eval_model(epoch, model, X, Y):
    Y_pred = model.forward(X)

    loss = optimizer.loss_fn(Y, Y_pred)
    mean_loss = mx.mean(mx.sum(loss, axis=0))

    errors = mx.sum(mx.abs(Y - mx.round(Y_pred)), axis=0)
    accuracy = mx.sum(errors == 0) / Y.shape[1]

    print(f"Epoch {epoch}: Accuracy {accuracy:.3f}, Average Loss {mean_loss}")

# batch gradient descent
MAX_EPOCHS = 5000
for epoch in range(MAX_EPOCHS):
    optimizer.train_network(train_x, train_y)
    if epoch % 100 == 0:
        eval_model(epoch, network, test_x, test_y)

eval_model(MAX_EPOCHS, network, test_x, test_y)
print("Finished")
