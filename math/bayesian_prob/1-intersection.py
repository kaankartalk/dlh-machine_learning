#!/usr/bin/env python3
"""Intersection module for Bayesian probability"""

import numpy as np
likelihood = __import__('0-likelihood').likelihood


def intersection(x, n, P, Pr):
    """Calculates the intersection of obtaining x and n with each
    probability in P.
    """
    # Validate n
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Validate x
    if not isinstance(x, int) or x < 0:
        msg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(msg)
    if x > n:
        raise ValueError("x cannot be greater than n")

    # Validate P
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Validate Pr
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Compute intersection: likelihood * prior
    like = likelihood(x, n, P)
    return like * Pr
