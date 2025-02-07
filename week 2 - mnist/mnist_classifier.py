import numpy as np
import pandas as pd
from src.network import Network

if __name__ == "__main__":
    train_csv = pd.read_csv('res/mnist_train.csv', header=None)
    test_csv = pd.read_csv('res/mnist_test.csv', header=None)

    def one_hot_encode(value, max_value):
        encoded = np.zeros(max_value + 1)
        encoded[value] = 1
        return encoded.reshape(-1, 1)

    train_set = []
    for index, row in train_csv.iterrows():
        train_set.append({
            'data': np.array(row[1:]).reshape(-1, 1) / 255,
            'label': one_hot_encode(row[0], 9)
        })

    test_set = []
    for index, row in test_csv.iterrows():
        test_set.append({
            'data': np.array(row[1:]).reshape(-1, 1) / 255,
            'label': one_hot_encode(row[0], 9)
        })

    print("MNIST Dataset Loaded!")

    network = Network(784, 10, [320, 160, 80, 40, 20], activation=Network.Activation.SIGMOID)
    network.train(train_set, batch_size=64, epochs=100, eta=3, validation_set=test_set)

    print("Network Trained!")