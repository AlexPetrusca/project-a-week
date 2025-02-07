import math
from src.dataset import DatasetGenerator
from src.network import Network

if __name__ == "__main__":
    datagen = DatasetGenerator(lambda x, y: int(x * math.sin(x) - y * math.cos(y) > 0))

    train_set = list(datagen.generate_samples(25000))
    test_set = list(datagen.generate_samples(1000))

    network = Network(784, 10, [30, 10])
    network.validate(test_set)
    network.train(train_set)
    network.validate(test_set, verbose=True)