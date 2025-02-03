import inspect
import numpy as np

class DatasetGenerator:
    def __init__(self, generator):
        self.generator = generator
        self.param_count = len(inspect.signature(generator).parameters)

    def generate_samples(self, n):
        for i in range(n):
            x = np.random.randn(self.param_count)
            y = self.generator(*x)
            yield {"data": x, "label": y}
