import random
import numpy as np

class Network:
    NULL = np.empty(0)

    def __init__(self, in_dim, out_dim, hid_dims):
        self.dims = [in_dim, *hid_dims, out_dim]
        self.weights = []
        self.biases = []
        for i in range(len(self.dims) - 1):
            num_neurons = self.dims[i + 1]
            num_weights = self.dims[i]
            self.weights.append(np.random.randn(num_neurons, num_weights))
            self.biases.append(np.random.randn(num_neurons, 1))

    def sigma(self, z):
        return 1 / (1 + np.e**(-z)) # sigmoid

    def sigma_prime(self, z):
        y = self.sigma(z)
        return y * (1 - y)

    def loss(self, y_true, y_pred):
        return (y_true - y_pred)**2 / 2 # squared error loss

    def loss_prime(self, y_true, y_pred):
        return -(y_true - y_pred) # squared error loss

    def feed_forward(self, x):
        for w, b in zip(self.weights, self.biases):
            z = w @ x + b  # weighted sum
            y = self.sigma(z)  # activation
            x = y  # output of this layer is input of the next
        return x

    def train(self, training_set, batch_size=32, epochs=10, eta=0.01, validation_set=None):
        def back_propagate(x, y):
            # feed forward
            z = [Network.NULL]
            a = [x]
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                z.append(w @ a[i] + b)  # weighted sum
                a.append(self.sigma(z[i + 1]))  # activation

            # backpropagate
            gradient_i = self.loss_prime(y, a[-1])
            for i in range(1, len(self.weights) + 1):
                if i == 1:
                    w_i = np.identity(gradient_i.shape[0])
                else:
                    w_i = self.weights[-i + 1].T

                gradient_i = (w_i @ gradient_i) * self.sigma_prime(z[-i])
                weight_gradient_i = gradient_i @ a[-i - 1].T
                bias_gradient_i = gradient_i @ np.ones((batch_size, 1))

                self.weights[-i] -= eta / batch_size * weight_gradient_i
                self.biases[-i] -= eta / batch_size * bias_gradient_i

        def create_batch(start):
            end = min(batch_start + batch_size, len(training_set))
            x = training_set[start]['data']
            y = training_set[start]['label']
            for idx in range(start + 1, end):
                x = np.hstack((x, training_set[idx]['data']))
                y = np.hstack((y, training_set[idx]['label']))
            return x, y

        def log_epoch(epoch):
            if validation_set is not None:
                validation_log = self.validate(validation_set)
                epoch_log = f"Epoch {epoch}/{epochs}:"
                print(f"{epoch_log:<15} {validation_log}")

        log_epoch(0)
        for epoch in range(epochs):
            random.shuffle(training_set)

            for batch_start in range(0, len(training_set), batch_size):
                x, y = create_batch(batch_start)
                if x.shape[1] == batch_size:
                    back_propagate(x, y)

            log_epoch(epoch + 1)

    def validate(self, validation_set, to_print=False):
        loss = 0
        accuracy = 0
        num_samples = 0
        for sample in validation_set:
            x = sample['data']
            y_pred = self.feed_forward(x)
            y_true = sample['label']

            num_samples += 1
            loss += self.loss(y_true, y_pred)
            if np.array_equal(np.round(y_pred), y_true):
                accuracy += 1

        accuracy /= num_samples
        loss = np.linalg.norm(loss)

        log = f"Accuracy: {accuracy:<10} Loss: {loss:<10}"
        if to_print:
            print(log)
        return log
