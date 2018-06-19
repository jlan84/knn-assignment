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

    Colors have been specifically chosen to be color blindness friendly.

    Assumes classifier, clf, has a .predict() method that follows the
    sci-kit learn functionality.

    X must contain only 2 continuous features.

    Function modeled on sci-kit learn example.

    Parameters
    ----------
    clf: instance of classifier object
        A fitted classifier with a .predict() method.
    X: numpy array, shape = [n_samples, n_features]
        Test data.
    y: numpy array, shape = [n_samples,]
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

    plt.title(
              "{0}-Class classification (k = {1}, metric = '{2}')"
              .format(n_classes, clf.k, clf.distance))
    plt.show()


def euclidean_distance(a, b):
    """Compute the euclidean_distance between two numpy arrays.

    Parameters
    ----------
    a: numpy array
    b: numpy array

    Returns
    -------
    numpy array
    """
    pass


def cosine_distance(a, b):
    """Compute the cosine_distance between two numpy arrays.

    Parameters
    ----------
    a: numpy array
    b: numpy array

    Returns
    -------
    """
    pass


class KNearestNeighbors(object):
    """Classifier implementing the k-nearest neighbors algorithm.

    Parameters
    ----------
    k: int, optional (default = 5)
        Number of neighbors that get a vote.
    distance: function, optional (default = euclidean)
        The distance function to use when computing distances.
    """

    def __init__(self, k=5, distance=euclidean_distance):
        """Initialize a KNearestNeighbors object."""
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        """Fit the model using X as training data and y as target labels.

        According to kNN algorithm, the training data is simply stored.

        Parameters
        ----------
        X: numpy array, shape = [n_samples, n_features]
            Training data.
        y: numpy array, shape = [n_samples,]
            Target labels.

        Returns
        -------
        None
        """
        pass

    def predict(self, X):
        """Return the predicted labels for the input X test data.

        Assumes shape of X is [n_test_samples, n_features] where n_features
        is the same as the n_features for the input training data.

        Parameters
        ----------
        X: numpy array, shape = [n_samples, n_features]
            Test data.

        Returns
        -------
        result: numpy array, shape = [n_samples,]
            Predicted labels for each test data sample.

        """
        pass

    def score(self, X, y_true):
        """Return the mean accuracy on the given data and true labels.

        Parameters
        ----------
        X: numpy array, shape = [n_samples, n_features]
            Test data.
        y_true: numpy array, shape = [n_samples,]
            True labels for given test data, X.

        Returns
        -------
        score: float
            Mean accuracy of self.predict(X) given true labels, y_true.
        """
        pass


if __name__ == '__main__':
    pass
