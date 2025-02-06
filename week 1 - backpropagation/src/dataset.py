import inspect
import numpy as np

class DatasetGenerator:
    def __init__(self, generator):
        self.generator = generator
        self.param_count = len(inspect.signature(generator).parameters)

    def generate_samples(self, n):
        for i in range(n):
            x = np.random.randn(self.param_count)
            y = np.array(self.generator(*x))
            yield {"data": x.reshape(-1, 1), "label": y.reshape(-1, 1)}
