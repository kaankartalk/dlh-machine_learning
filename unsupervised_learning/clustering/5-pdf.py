#!/usr/bin/env python3
"""Calculates the probability density function of a Gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """Calculates the probability density function of a Gaussian distribution"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape
    if m.shape[0] != d or S.shape != (d, d):
        return None

    X_m = X - m
    inv_S = np.linalg.inv(S)
    det_S = np.linalg.det(S)

    exponent = np.einsum('ij,jk,ik->i', X_m, inv_S, X_m)
    denom = np.sqrt(((2 * np.pi) ** d) * det_S)
    P = np.exp(-0.5 * exponent) / denom

    P = np.maximum(P, 1e-300)

    return P
