import numpy as np

class Network:

    def __init__(self, in_dim, out_dim, hid_dims):
        self.dims = [in_dim, *hid_dims, out_dim]
        self.eta = 0.01  # learning rate

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
        return -(y_true - y_pred) # squared error loss

    def feed_forward(self, *x):
        # a = [x] * (len(self.weights) + 1)
        # for i, layer in enumerate(self.weights):
        #     a[i] = np.append(a[i], 1)  # append bias term
        #     z = layer.dot(a[i])  # weighted sum
        #     y = self.sigma(z)  # activation
        #     a[i + 1] = y  # output of this layer is input of the next

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

    def train_0(self, test_set):
        for sample in test_set:
            # read sample
            x = sample['data']
            y_true = sample['label']

            # feed forward
            a_0 = np.append(x, 1)
            a_1 = self.sigma(self.weights[0].dot(a_0))

            # create X matrix: each row is a copy of x
            A_0 = np.zeros(self.weights[0].shape)
            A_0[:] = a_0

            # subtract gradient from layer's weights
            gradient = self.loss_prime(y_true, a_1) *  self.sigma_prime(a_1)
            self.weights[0] -= self.eta * gradient * A_0

    def train_1(self, test_set):
        for sample in test_set:
            # read sample
            x = sample['data']
            y_true = sample['label']

            # feed forward
            a_0 = np.append(x, 1)
            a_1 = np.append(self.sigma(self.weights[0].dot(a_0)), 1)
            a_2 = self.sigma(self.weights[1].dot(a_0))

            # backpropagate
            A_0 = np.zeros(self.weights[0].shape)
            A_0[:] = a_0

            A_1 = np.zeros(self.weights[1].shape)
            A_1[:] = a_1

            gradient_1 = self.loss_prime(y_true, a_2) * self.sigma_prime(a_2)
            gradient_0 = np.dot(gradient_1, self.weights[1]) * self.sigma_prime(a_1)

            self.weights[1] -= self.eta * gradient_1 * A_1
            self.weights[0] -= self.eta * gradient_0 * A_0

    def train_n(self, test_set):
        pass

    def validate(self, validation_set, verbose=False, print_samples=10):
        average_loss = 0
        accuracy = 0
        num_samples = 0
        for sample in validation_set:
            x = sample['data']
            y_pred = self.feed_forward(sample['data'])
            y_true = sample['label']

            loss = self.loss(y_true, y_pred)
            if verbose and num_samples < print_samples:
                print(f"Data: {str(x):<35} Label: {y_true:<5} Prediction: {y_pred:<24} Loss: {loss:<24}")

            num_samples += 1
            average_loss += loss
            if round(y_pred) == y_true:
                accuracy += 1

        accuracy /= num_samples
        average_loss /= num_samples
        final_log = f"Accuracy: {accuracy:<10} Average Loss: {average_loss}"
        if verbose:
            print(f"{final_log:>117}")
        else:
            print(f"{final_log}")
