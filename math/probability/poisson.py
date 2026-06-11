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

        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        def factorial(n):
            result = 1.0
            for i in range(1, n + 1):
                result *= i
            return result

        def exp(x):
            total = 1.0
            term = 1.0
            i = 1
            # keep adding terms until they are very small
            while True:
                term *= x / i
                if term == 0 or abs(term) < 1e-15:
                    break
                total += term
                i += 1
            return total

        lambtha = self.lambtha

        return (exp(-lambtha) * (lambtha ** k)) / factorial(k)
