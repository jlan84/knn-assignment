import numpy as np
import matplotlib.pyplot as plt
from jl_basic import *
from make_data import make_data
from sklearn.metrics import mean_squared_error


def plot_predictions(ax, reg, X, y):
    """Plot the decision boundary of a kNN regressor.

    Plots predictions as colors.

    Assumes regressor, reg, has a .predict() method that follows the
    sci-kit learn functionality.

    X must contain only 2 continuous features.

    Function modeled on scikit-learn example.

    Colors have been chosen for accessibility.


    Parameters
    ----------
    reg: instance of regressor object
        A fitted regressor with a .predict() method.
    X: numpy array, shape = [n_observations, n_features]
        Training data to display.
    y: numpy array, shape = [n_observations,]
        Target labels.
    """
    mesh_count = 100.

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    feature_1 = X[:, 0]
    feature_2 = X[:, 1]
    x_min, x_max = feature_1.min(), feature_1.max()
    y_min, y_max = feature_2.min(), feature_2.max()
    v_min, v_max = y.min(), y.max()

    x_mesh_step_size = (x_max - x_min)/mesh_count
    y_mesh_step_size = (y_max - y_min)/mesh_count

    xx, yy = np.meshgrid(np.arange(x_min, x_max, x_mesh_step_size),
                         np.arange(y_min, y_max, y_mesh_step_size))
    values = reg.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    values = values.reshape(xx.shape)
    ax.pcolormesh(xx, yy,
                  values,
                  cmap='viridis',
                  vmin=v_min,
                  vmax=v_max)

    # Plot the training points, saving the colormap
    sctr = ax.scatter(feature_1, feature_2,
                      c=y,
                      cmap='viridis',
                      edgecolor='black', lw=0.2)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    ax.set_title("Regression predictions (k = {0}, metric = '{1}')"
                 .format(reg.k, reg.distance.__name__))

    

if __name__ == "__main__":
    X, y = make_data(n_features=2, n_pts=300, noise=0.1)
    knn = KNNRegressor(k=1, distance=manhattan_distance)
    knn.fit(X,y)
    y_pred = knn.predict(X)
    print(mean_squared_error(y,y_pred))
    fig, ax = plt.subplots()
    plot_predictions(ax, knn, X, y)
    plt.show()
    
