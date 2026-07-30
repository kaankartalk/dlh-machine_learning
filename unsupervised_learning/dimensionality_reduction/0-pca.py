#!/usr/bin/env python3
"""Principal Component Analysis on a zero-mean dataset."""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d), all dimensions have mean 0.
        var: fraction of the variance the transformation should maintain.

    Returns:
        W: numpy.ndarray of shape (d, nd), the weights matrix.
    """
    _, S, Vt = np.linalg.svd(X)
    ratios = np.cumsum(S) / np.sum(S)
    nd = np.argwhere(ratios >= var)[0, 0]
    return Vt[:nd + 1].T