from src.dataset import DatasetGenerator
from src.network import Network

if __name__ == "__main__":
    datagen = DatasetGenerator(lambda x, y: int(x + y > 0)) # sum is positive
    # datagen = DatasetGenerator(lambda x, y: int(5 * x + 5 * y > 3))  # linear boundary
    train_set = list(datagen.generate_samples(100000))
    test_set = list(datagen.generate_samples(1000))

    network = Network(2, 1, [])
    network.validate(test_set)
    network.train(train_set)
    network.validate(test_set, verbose=True)