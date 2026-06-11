#!/usr/bin/env python3
"""Poisson distribution class"""


class Poisson:
    """Represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Class constructor"""

        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculates PMF for given number of successes k"""

        # Convert non-integer k to int
        if not isinstance(k, int):
            k = int(k)

        # PMF is zero for negative k
        if k < 0:
            return 0

        # Manual factorial
        def factorial(n):
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result

        # Manual exponential (Taylor series, 100 terms for precision)
        def exp(x):
            total = 1
            term = 1
            for i in range(1, 100):
                term *= x / i
                total += term
            return total

        λ = self.lambtha

        return (exp(-λ) * (λ ** k)) / factorial(k)
