#!/usr/bin/env python3
"""Cost of the t-SNE transformation."""
import numpy as np


def cost(P, Q):
    """
    Calculates the cost of the t-SNE transformation.

    Args:
        P: numpy.ndarray of shape (n, n) with the P affinities.
        Q: numpy.ndarray of shape (n, n) with the Q affinities.

    Returns:
        C: the cost of the transformation.
    """
    P = np.where(P != 0, P, 1e-12)
    Q = np.where(Q != 0, Q, 1e-12)
    return np.sum(P * np.log(P / Q))
