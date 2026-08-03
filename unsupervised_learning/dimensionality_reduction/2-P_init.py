#!/usr/bin/env python3
"""Initializes the variables required to calculate the P affinities."""
import numpy as np


def P_init(X, perplexity):
    """
    Initializes all variables required to calculate the P affinities.

    Args:
        X: numpy.ndarray of shape (n, d) to be transformed by t-SNE.
        perplexity: the perplexity all Gaussian distributions should have.

    Returns:
        (D, P, betas, H)
    """
    n = X.shape[0]
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.matmul(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return D, P, betas, H