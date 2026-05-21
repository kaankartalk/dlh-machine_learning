#!/usr/bin/env python3
"""
Module for determining the definiteness of a matrix.
"""
import numpy as np


def definiteness(matrix):
    """
    Determines the definiteness of a square matrix.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.size == 0 or len(matrix.shape) != 2:
        return None
    n, m = matrix.shape
    if n != m:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    try:
        vals = np.linalg.eigvals(matrix)
    except Exception:
        return None

    vals = np.real(vals)

    if np.all(vals > 0):
        return "Positive definite"
    if np.all(vals >= 0):
        return "Positive semi-definite"
    if np.all(vals < 0):
        return "Negative definite"
    if np.all(vals <= 0):
        return "Negative semi-definite"
    if np.any(vals > 0) and np.any(vals < 0):
        return "Indefinite"

    return None
