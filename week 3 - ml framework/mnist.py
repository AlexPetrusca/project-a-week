import pandas as pd
import mlx.core as mx

train_csv = pd.read_csv('res/mnist_train.csv', header=None)
test_csv = pd.read_csv('res/mnist_test.csv', header=None)

def one_hot_encode(value, max_value):
    encoded = mx.zeros(max_value + 1)
    encoded[value] = 1
    return encoded.reshape(-1, 1)

train_images = mx.empty((2, 0))
train_labels = []
for index, row in train_csv.iterrows():
    train_images.append(mx.array(row[1:]).reshape(-1, 1) / 255)
    train_labels.append(one_hot_encode(row[0], 9))

test_images = []
test_labels = []
for index, row in test_csv.iterrows():
    test_images.append(mx.array(row[1:]).reshape(-1, 1) / 255)
    test_labels.append(one_hot_encode(row[0], 9))