import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import sys


def plot_decision_boundary(clf, X, y, n_classes):
    """Plot the decision boundary of a kNN classifier.

    Plots decision boundary for up to 4 classes.

    Assumes classifier, clf, has a .predict() method that follows the
    sci-kit learn functionality.

    X must contain only 2 continuous features.

    Function modeled on scikit-learn example.

    Colors have been chosen for accessibility.


    Parameters
    ----------
    clf: instance of classifier object
        A fitted classifier with a .predict() method.
    X: numpy array, shape = [n_observations, n_features]
        Test data.
    y: numpy array, shape = [n_observations,]
        Target labels.
    n_classes: int
        The number of classes in the target labels.
    """
    mesh_step_size = .1

    # Colors are in the order [red, yellow, blue, cyan]
    cmap_light = ListedColormap(['#FFAAAA', '#FFFFAA', '#AAAAFF', '#AAFFFF'])
    cmap_bold = ListedColormap(['#FF0000', '#FFFF00', '#0000FF', '#00CCCC'])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    feature_1 = X[:, 0]
    feature_2 = X[:, 1]
    x_min, x_max = feature_1.min() - 1, feature_1.max() + 1
    y_min, y_max = feature_2.min() - 1, feature_2.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))
    dec_boundary = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    dec_boundary = dec_boundary.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, dec_boundary, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(feature_1, feature_2, c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.title("{0}-Class classification (k = {1}, metric = '{2}')"
              .format(n_classes, clf.k, clf.distance))
    plt.show()

