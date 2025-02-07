import math
from src.dataset import DatasetGenerator
from src.network import Network

if __name__ == "__main__":
    datagen = DatasetGenerator(lambda x, y: int(x * math.sin(x) - y * math.cos(y) > 0))

    train_set = list(datagen.generate_samples(100000))
    validation_set = list(datagen.generate_samples(10000))
    test_set = list(datagen.generate_samples(10000))

    network = Network(2, 1, [30, 10])
    network.validate(test_set)
    network.train(train_set, batch_size=100, epochs=10, eta=0.5, validation_set=validation_set)