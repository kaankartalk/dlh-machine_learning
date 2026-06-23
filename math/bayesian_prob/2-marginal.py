#!/usr/bin/env python3
import numpy as np


def marginal(x, n, P, Pr):
    """Calculates the marginal probability of obtaining the data"""

    # 1) Validate n
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # 2) Validate x
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    # 3) x cannot exceed n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # 4) Validate P
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # 5) Validate Pr shape
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    # 6) Validate P values
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    # 7) Validate Pr values
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    # 8) Pr must sum to 1
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Compute likelihood
    # Binomial PMF: C(n,x) * p^x * (1-p)^(n-x)
    fact = np.math.factorial
    comb = fact(n) / (fact(x) * fact(n - x))
    likelihood = comb * (P ** x) * ((1 - P) ** (n - x))

    # Intersection = likelihood * prior
    intersection = likelihood * Pr

    # Marginal = sum of intersections
    return np.sum(intersection)
