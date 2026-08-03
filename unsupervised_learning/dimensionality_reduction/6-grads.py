#!/usr/bin/env python3
"""Gradients of the low dimensional map Y in t-SNE."""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates the gradients of Y.

    Args:
        Y: numpy.ndarray of shape (n, ndim) with the low dimensional
            transformation of X.
        P: numpy.ndarray of shape (n, n) with the P affinities of X.

    Returns:
        (dY, Q)
    """
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)
    dY = np.zeros((n, ndim))
    PQ = (P - Q) * num

    for i in range(n):
        dY[i] = np.sum(PQ[i][:, None] * (Y[i] - Y), axis=0)

    return dY, Q
