import gzip
import os
import pickle
from PIL import Image
from urllib import request

import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt

from alpineml import Network
from alpineml.layer.reshape import Flatten, Reshape
from alpineml.optim import SGD
from alpineml.function.activation import leaky_relu, softmax
from alpineml.function.loss import CrossEntropyLoss, cross_entropy_loss
from alpineml.layer.core import Linear, Activation


# Load Dataset
def mnist(
    save_dir="/tmp",
    base_url="https://raw.githubusercontent.com/fgnt/mnist/master/",
    filename="mnist_2d.pkl",
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
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
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
        return x.astype(np.float32) / 255.0  # x: [0, 255] --> [0, 1]
        # return 2 * (x.astype(np.float32) / 255.0) - 1  # x: [0, 255] --> [-1, 1]

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
        mnist["training_images"],
        mnist["training_labels"],
        mnist["test_images"],
        mnist["test_labels"],
    )


def fashion_mnist(save_dir="/tmp"):
    return mnist(
        save_dir,
        base_url="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        filename="fashion_mnist_2d.pkl",
    )


# Visualize
def viz_sample_predictions(X, Y_true, Y_pred, label_map, rows=5, cols=5, figsize=(10, 10)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize, num="Sample Predictions")
    axes = axes.reshape(-1)  # flatten

    def sample_random():
        for j in np.arange(0, rows * cols):
            i = np.random.randint(0, Y_true.shape[0])

            raw_sample = X[i].reshape(28, 28)
            sample = np.array(255 * raw_sample)
            image = Image.fromarray(sample)

            raw_label = mx.argmax(Y_true[i]).item()
            label = label_map[raw_label]

            raw_pred = mx.argmax(Y_pred[i]).item()
            pred = label_map[raw_pred]

            axes[j].imshow(image)
            axes[j].set_title(f"True: {label} \nPredict: {pred}")
            axes[j].axis('off')
            plt.subplots_adjust(wspace=1)

    def on_key(event):
        if event.key == ' ':
            sample_random()
            fig.show()

    fig.canvas.mpl_connect('key_press_event', on_key)

    sample_random()


def viz_history(history, figsize=(6, 4)):
    plt.figure(figsize=figsize, num="Loss Curves")
    plt.plot(history['epoch'], history['train_loss'], 'black', linewidth=2.0)
    plt.plot(history['epoch'], history['test_loss'], 'green', linewidth=2.0)
    plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.title('Loss vs Epoch', fontsize=12)

    plt.figure(figsize=figsize, num="Accuracy Curves")
    plt.plot(history['epoch'], history['train_accuracy'], 'black', linewidth=2.0)
    plt.plot(history['epoch'], history['test_accuracy'], 'green', linewidth=2.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.title('Accuracy vs Epoch', fontsize=12)


# Evaluate
def eval_model(model, X, Y, epoch=None):
    Y_pred = model.forward(X)

    loss = optimizer.loss_fn(Y_pred, Y)
    mean_loss = mx.mean(mx.sum(loss, axis=1))

    if isinstance(optimizer.loss_fn, CrossEntropyLoss):
        Y_pred = softmax(Y_pred)

    errors = mx.sum(mx.abs(Y - mx.round(Y_pred)), axis=1)
    accuracy = mx.sum(errors == 0) / Y.shape[0]

    if epoch is not None:
        print(f"Epoch {epoch}: Accuracy {accuracy:.3f}, Average Loss {mean_loss}")
    else:
        print(f"Accuracy {accuracy:.3f}, Average Loss {mean_loss}")
    return Y_pred, accuracy, mean_loss


def train(train_x, train_y, epochs, batch_size=1, dilation=1, test_x=None, test_y=None, cb=None):
    # batch input
    train_x_dilated = mx.repeat(train_x, dilation, axis=0)
    train_y_dilated = mx.repeat(train_y, dilation, axis=0)
    train_x_batched = mx.split(train_x_dilated, int(train_x_dilated.shape[0] / batch_size), axis=0)
    train_y_batched = mx.split(train_y_dilated, int(train_y_dilated.shape[0] / batch_size), axis=0)

    history = {"epoch": [], "train_loss": [], "test_loss": [], "train_accuracy": [], "test_accuracy": []}
    for epoch in range(1, epochs + 1):
        for x_batch, y_batch in zip(train_x_batched, train_y_batched):
            optimizer.step(x_batch, y_batch)

        _, train_accuracy, train_loss = eval_model(network, train_x, train_y, epoch=epoch)
        _, test_accuracy, test_loss = eval_model(network, test_x, test_y, epoch=epoch)
        print()
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_accuracy"].append(test_accuracy)

    _, train_accuracy, train_loss = eval_model(network, train_x, train_y, epoch=epochs)
    _, test_accuracy, test_loss = eval_model(network, test_x, test_y, epoch=epochs)
    print()
    history["epoch"].append(epochs)
    history["train_loss"].append(train_loss)
    history["test_loss"].append(test_loss)
    history["train_accuracy"].append(train_accuracy)
    history["test_accuracy"].append(test_accuracy)

    pred_y, _, _ = eval_model(network, test_x, test_y)
    print()

    viz_sample_predictions(test_x, test_y, pred_y, label_map)
    viz_history(history)
    plt.show()


train_x, train_y, test_x, test_y = map(mx.array, mnist()) # 97% max accuracy
label_map = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

# train_x, train_y, test_x, test_y = map(mx.array, fashion_mnist()) # 87% max accuracy
# label_map = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

network = Network(input_shape=(28, 28))
network.add_layer(Flatten())
network.add_layer(Linear(320))
network.add_layer(Activation(leaky_relu))
network.add_layer(Linear(160))
network.add_layer(Activation(leaky_relu))
network.add_layer(Linear(80))
network.add_layer(Activation(leaky_relu))
network.add_layer(Linear(10))
network.add_layer(Activation(leaky_relu))

optimizer = SGD(eta=0.1, momentum=0.9, weight_decay=0.0005)
optimizer.bind_loss_fn(cross_entropy_loss)
optimizer.bind_network(network)

train(train_x, train_y, epochs=25, batch_size=1000, dilation=1, test_x=test_x, test_y=test_y)
