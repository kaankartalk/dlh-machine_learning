#!/usr/bin/env python3
"""Q affinities for the low dimensional map in t-SNE."""
import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affinities.

    Args:
        Y: numpy.ndarray of shape (n, ndim) with the low dimensional
            transformation of X.

    Returns:
        (Q, num)
    """
    sum_Y = np.sum(np.square(Y), axis=1)
    D = np.add(np.add(-2 * np.matmul(Y, Y.T), sum_Y).T, sum_Y)
    num = 1 / (1 + D)
    np.fill_diagonal(num, 0)
    Q = num / np.sum(num)
    return Q, num
