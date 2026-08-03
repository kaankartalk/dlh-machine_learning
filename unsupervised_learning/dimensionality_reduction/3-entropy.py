#!/usr/bin/env python3
"""Shannon entropy and P affinities relative to a data point."""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a point.

    Args:
        Di: numpy.ndarray of shape (n - 1,) with the pairwise distances
            between a data point and all other points except itself.
        beta: numpy.ndarray of shape (1,) with the beta value for the
            Gaussian distribution.

    Returns:
        (Hi, Pi)
    """
    num = np.exp(-Di * beta)
    Pi = num / np.sum(num)
    Hi = -np.sum(Pi * np.log2(Pi))
    return Hi, Pi
