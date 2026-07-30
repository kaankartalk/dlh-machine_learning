#!/usr/bin/env python3
"""Performs K-means on a dataset"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    low = X.min(axis=0)
    high = X.max(axis=0)
    C = np.random.uniform(low, high, (k, d))

    for i in range(iterations):
        C_prev = np.copy(C)

        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        for j in range(k):
            if np.sum(clss == j) == 0:
                C[j] = np.random.uniform(low, high, d)
            else:
                C[j] = X[clss == j].mean(axis=0)

        if np.array_equal(C, C_prev):
            return C, clss

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)
    return C, clss
