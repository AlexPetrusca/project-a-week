import numpy as np

from src.dataset import DatasetGenerator
from src.network import Network

if __name__ == "__main__":
    # datagen = DatasetGenerator(lambda x, y: np.repeat(int(x + y > 0), 4)) # sum is positive
    # datagen = DatasetGenerator(lambda x, y: int(5 * x + 5 * y > 3))  # linear boundary

    datagen = DatasetGenerator(lambda x, y: int(bool(x > 0) & bool(y > 0))) # logical and

    train_set = list(datagen.generate_samples(100000))
    test_set = list(datagen.generate_samples(1000))

    network = Network(2, 1, [])
    network.validate(test_set)
    network.train_0(train_set)
    network.validate(test_set, verbose=True)