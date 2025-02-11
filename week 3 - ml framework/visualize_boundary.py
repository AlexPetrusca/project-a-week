import numpy as np
import matplotlib.pyplot as plt
from src.network import Network

if __name__ == "__main__":
    # Fetch the dataset
    train_set = []
    pts = np.loadtxt('./res/points.txt')
    X, Y = pts[:, :2], pts[:, 2:]
    for x, y in zip(X, Y):
        train_set.append({'data': x.reshape(-1, 1), 'label': y.reshape(-1, 1)})

    # Create the model
    network = Network(2, 1, [16])

    # Define plotting function
    plt.ion()

    def callback(ctx):
        log_validation_error(ctx)
        plot_decision_boundary(ctx)

    def log_validation_error(ctx):
        validation_set = ctx['validation_set']
        if validation_set is not None:
            validation_log = ctx['network'].validate(validation_set)
            epoch_log = f"Epoch {ctx['epoch']}/{ctx['epochs']}:"
            print(f"{epoch_log:<15} {validation_log}")

    def plot_decision_boundary(ctx):
        # clear
        plt.clf()
        # Set min and max values and give it some padding.
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        h = 0.01
        # Generate a meshgrid of points with orthogonal spacing.
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Prediction of the classified values across the whole grid.
        Z = np.round(ctx['network'].feed_forward(np.c_[xx.ravel(), yy.ravel()].T)[-1])
        Z = Z.reshape(xx.shape)
        # Plot the decision boundary as a contour plot and training examples.
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=Y.T, cmap=plt.cm.Spectral, s=8)
        # update axes
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # stop or pause
        if ctx.get('status') == 'Done':
            plt.show(block=True)
        else:
            plt.pause(1 / 1000)

    # Train the model
    network.train(
        train_set,
        validation_set=train_set,
        batch_size=pts.shape[0], epochs=10000, eta=2,
        epoch_cb=callback, epoch_cb_stride=100
    )
