import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt

from alpineml import Network, Optimizer
from alpineml.layer import Linear, Activation
from alpineml.function.loss import MSELoss, BinaryCrossEntropyLoss
from alpineml.function.activation import Sigmoid, Relu


def plot_decision_boundary(points, labels):
    # Set min and max values and give it some padding.
    x_min, x_max = points[0].min() - 0.1, points[0].max() + 0.1
    y_min, y_max = points[1].min() - 0.1, points[1].max() + 0.1
    h = 0.01
    # Generate a meshgrid of points with orthogonal spacing.
    xx, yy = mx.meshgrid(mx.arange(x_min.item(), x_max.item(), h), mx.arange(y_min.item(), y_max.item(), h))
    nxx, nyy = xx.flatten().reshape(-1, 1), yy.flatten().reshape(-1, 1)
    # Prediction of the classified values across the whole grid.
    grid_points = mx.concatenate([nxx, nyy], axis=1).T
    grid_predict = network.forward(grid_points)
    Z = mx.round(grid_predict).reshape(xx.shape)
    # Plot the decision boundary as a contour plot and training examples.
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(points[0, :], points[1, :], c=labels, cmap=plt.cm.Spectral, s=8)
    # update axes
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.pause(0.0001)

network = Network()
network.add_layer(Linear(2, 16))
network.add_layer(Activation(Relu()))
network.add_layer(Linear(16, 8))
network.add_layer(Activation(Relu()))
network.add_layer(Linear(8, 1))
network.add_layer(Activation(Sigmoid()))

optimizer = Optimizer()
optimizer.bind_network(network)
optimizer.bind_loss_fn(BinaryCrossEntropyLoss())
optimizer.bind_learning_rate(0.5)

pts = np.loadtxt('res/points.txt')
X, Y = mx.array(pts[:, :2].T), mx.array(pts[:, 2:].T)

def eval_model(epoch, model, X, Y):
    Y_pred = model.forward(X)

    loss = optimizer.loss_fn(Y, Y_pred)
    mean_loss = mx.mean(mx.sum(loss, axis=0))

    errors = mx.sum(mx.abs(Y - mx.round(Y_pred)), axis=0)
    accuracy = mx.sum(errors == 0) / Y.shape[1]

    print(f"Epoch {epoch}: Accuracy {accuracy:.3f}, Average Loss {mean_loss}")

for i in range(50000):
    optimizer.train_network(X, Y)
    if i % 1000 == 0:
        eval_model(i, network, X, Y)
        plot_decision_boundary(X, Y)

plt.show(block=True)