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

    def train_0(self, test_set):
        for sample in test_set:
            # read sample
            x = sample['data']
            y_true = sample['label']

            # feed forward
            a_0 = x
            z_1 = self.weights[0] @ a_0 + self.biases[0]
            a_1 = self.sigma(z_1)

            # subtract gradient from layer's weights
            gradient_i = self.loss_prime(y_true, a_1) * self.sigma_prime(z_1)
            weight_gradient_i = gradient_i @ a_0.T
            self.weights[0] -= self.eta * weight_gradient_i
            self.biases[0] -= self.eta * gradient_i

    def train_1(self, test_set):
        for sample in test_set:
            # read sample
            x = sample['data']
            y_true = sample['label']

            # feed forward
            a_0 = x
            z_1 = self.weights[0] @ a_0 + self.biases[0]
            a_1 = self.sigma(z_1)
            z_2 = (self.weights[1] @ a_1 + self.biases[1])
            a_2 = self.sigma(z_2)

            # backpropagate
            gradient_2 = self.loss_prime(y_true, a_2) * self.sigma_prime(z_2)
            weight_gradient_2 = gradient_2 @ a_1.T
            self.weights[1] -= self.eta * weight_gradient_2
            self.biases[1] -= self.eta * gradient_2

            gradient_1 = (self.weights[1].T @ gradient_2) * self.sigma_prime(z_1)
            weight_gradient_1 = gradient_1 @ a_0.T
            self.weights[0] -= self.eta * weight_gradient_1
            self.biases[0] -= self.eta * gradient_1

    def train_n(self, test_set):
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
            gradient_i = self.loss_prime(y_true, a[-1]) * self.sigma_prime(z[-1])
            weight_gradient_i = gradient_i @ a[-2].T
            self.weights[-1] -= self.eta * weight_gradient_i
            self.biases[-1] -= self.eta * gradient_i
            for i in range(2, len(self.weights) + 1):
                gradient_i = (self.weights[-i + 1].T @ gradient_i) * self.sigma_prime(z[-i])
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
            # if verbose and num_samples < print_samples:
                # print(f"Data: {str(x):<35} Label: {y_true:<5} Prediction: {y_pred:<24} Loss: {loss:<24}")

            num_samples += 1
            average_loss += loss
            if np.array_equal(np.round(y_pred), y_true):
                accuracy += 1

        accuracy /= num_samples
        average_loss /= num_samples
        final_log = f"Accuracy: {accuracy:<10} Average Loss: {average_loss}"
        if verbose:
            print(f"{final_log:>117}")
        else:
            print(f"{final_log}")
