#!/usr/bin/env python3
"""Calculates the maximization step in the EM algorithm for a GMM"""
import numpy as np


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape
    if n_g != n:
        return None, None, None
    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None

    Nk = np.sum(g, axis=1)

    pi = Nk / n
    m = np.matmul(g, X) / Nk[:, np.newaxis]

    S = np.zeros((k, d, d))
    for i in range(k):
        X_m = X - m[i]
        S[i] = np.matmul(g[i] * X_m.T, X_m) / Nk[i]

    return pi, m, S
