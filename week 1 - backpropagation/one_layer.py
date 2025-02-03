from src.dataset import DatasetGenerator
from src.network import Network

if __name__ == "__main__":
    datagen = DatasetGenerator(lambda x, y: int(x) + int(y)) # addition
    # datagen = DatasetGenerator(lambda x, y: int(bool(x > 0) & bool(y > 0))) # logical and
    # datagen = DatasetGenerator(lambda x, y: int(bool(x > 0) ^ bool(y > 0))) # logical xor
    # datagen = DatasetGenerator(lambda x, y: int(5 * x**2 + 5 * y**2 > 3)) # quadratic boundary
    train_set = list(datagen.generate_samples(100000))
    test_set = list(datagen.generate_samples(1000))

    network = Network(2, 1, [2])
    network.validate(test_set)
    network.train(train_set)
    network.validate(test_set, verbose=True)