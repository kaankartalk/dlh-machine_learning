#!/usr/bin/env python3
"""Principal Component Analysis with a fixed output dimensionality."""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        ndim: the new dimensionality of the transformed X.

    Returns:
        T: numpy.ndarray of shape (n, ndim), the transformed X.
    """
    X_m = X - np.mean(X, axis=0)
    _, _, Vt = np.linalg.svd(X_m)
    W = Vt[:ndim].T
    return np.matmul(X_m, W)