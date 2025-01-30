import numpy as np


def euclidean_distance(X, Y):
    products = X @ Y.T
    x_norms = np.sum(X * X, axis=1)
    x_norms = np.repeat(x_norms[..., np.newaxis], Y.shape[0], axis=1)
    y_norms = np.sum(Y * Y, axis=1)
    y_norms = np.repeat(y_norms[..., np.newaxis], X.shape[0], axis=1).T
    return (x_norms - 2 * products + y_norms) ** 0.5


def cosine_distance(X, Y):
    products = (X @ Y.T)
    x_norms = np.sum(X * X, axis=1)
    x_norms = np.repeat(x_norms[..., np.newaxis], Y.shape[0], axis=1) ** 0.5
    y_norms = np.sum(Y * Y, axis=1)
    y_norms = np.repeat(y_norms[..., np.newaxis], X.shape[0], axis=1).T ** 0.5
    return 1 - products / x_norms / y_norms
