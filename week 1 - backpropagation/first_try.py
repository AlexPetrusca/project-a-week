import numpy as np
import inspect

class DatasetGenerator:
    def __init__(self, generator):
        self.generator = generator
        self.param_count = len(inspect.signature(generator).parameters)

    def generate_samples(self, n):
        for i in range(n):
            x = np.random.randn(self.param_count)
            y = self.generator(*x)
            yield {"data": x, "label": int(y)}

class Network:
    ETA = 1 # learning rate

    def __init__(self, in_dim, out_dim, hid_dims):
        self.dims = [in_dim, *hid_dims, out_dim]

        self.weights = []
        for i in range(len(self.dims) - 1):
            num_neurons = self.dims[i + 1]
            num_weights = self.dims[i] + 1
            self.weights.append(np.random.randn(num_neurons, num_weights))

    def sigma(self, z):
        return 1 / (1 + np.e**(-z)) # sigmoid

    def sigma_prime(self, z):
        y = self.sigma(z)
        return y * (1 - y)

    def loss(self, y_true, y_pred):
        return (y_true - y_pred)**2 / 2 # squared error loss

    def loss_prime(self, y_true, y_pred):
        return y_true - y_pred # squared error loss

    def feed_forward(self, *x):
        def _feed_forward(a):
            for layer in self.weights:
                a = np.append(a, 1) # append bias term
                z = layer.dot(a) # weighted sum
                y = self.sigma(z) # activation
                a = y # output of this layer is input of the next
            return a

        y = _feed_forward(x)
        if len(y) == 1:
            return y[0] # unwrap scalar value
        else:
            return y # return output vector

    def train(self, test_set, batch_size=32):
        pass

    def validate(self, validation_set):
        batch_loss = 0
        for sample in validation_set:
            x = sample['data']
            y_pred = network.feed_forward(sample['data'])
            y_true = sample['label']
            loss = network.loss(y_true, y_pred)
            batch_loss += loss
            print(f"Data: {str(x):<35} Label: {y_true:<5} Prediction: {y_pred:<24} Loss: {loss:<24}")
        print(f"{'':>86}Total Loss: {batch_loss}")

def logical_and(x, y):
    return x > 0 and y > 0

datagen = DatasetGenerator(logical_and)
train_dataset = datagen.generate_samples(10000)
test_dataset = datagen.generate_samples(100)

network = Network(2, 1, [5, 4])
network.train(train_dataset)
network.validate(test_dataset)