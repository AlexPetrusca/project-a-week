import numpy as np

class Network:
    NULL = np.empty(0)

    def __init__(self, in_dim, out_dim, hid_dims):
        self.dims = [in_dim, *hid_dims, out_dim]
        self.eta = 0.01  # learning rate

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

    def train(self, test_set):
        for sample in test_set:
            # read sample
            x = sample['data']
            y_true = sample['label']

            # feed forward
            z = [Network.NULL]
            a = [x]
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                z.append(w @ a[i] + b)  # weighted sum
                a.append(self.sigma(z[i + 1]))  # activation

            # backpropagate
            gradient_i = self.loss_prime(y_true, a[-1])
            for i in range(1, len(self.weights) + 1):
                if i == 1:
                    w_i = np.identity(gradient_i.shape[0])
                else:
                    w_i = self.weights[-i + 1].T

                gradient_i = (w_i @ gradient_i) * self.sigma_prime(z[-i])
                weight_gradient_i = gradient_i @ a[-i - 1].T
                self.weights[-i] -= self.eta * weight_gradient_i
                self.biases[-i] -= self.eta * gradient_i

    def validate(self, validation_set, verbose=False, print_samples=10):
        average_loss = 0
        accuracy = 0
        num_samples = 0
        for sample in validation_set:
            x = sample['data']
            y_pred = self.feed_forward(x)
            y_true = sample['label']

            loss = self.loss(y_true, y_pred)

            num_samples += 1
            average_loss += loss
            if np.array_equal(np.round(y_pred), y_true):
                accuracy += 1

        accuracy /= num_samples
        average_loss /= num_samples
        print(f"Accuracy: {accuracy:<10} Average Loss: {average_loss}")
