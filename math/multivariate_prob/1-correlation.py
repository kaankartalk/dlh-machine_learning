#!/usr/bin/env python3
import numpy as np

def mean_cov(X):
    """Calculates the mean and covariance of a data set"""

    # Validate input
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Mean: shape (1, d)
    mean = np.mean(X, axis=0, keepdims=True)

    # Center data
    X_centered = X - mean

    # Covariance: (1/(n-1)) * X_centered^T * X_centered
    cov = (X_centered.T @ X_centered) / (n - 1)

    return mean, cov
