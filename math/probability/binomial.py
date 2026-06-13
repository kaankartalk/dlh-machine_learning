#!/usr/bin/env python3
"""Binomial distribution class"""


class Binomial:
    """Represents a Binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Class constructor"""

        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = int(n)
            self.p = float(p)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            var = sum((x - mean) ** 2 for x in data) / len(data)

            # Method of Moments
            p_est = 1 - (var / mean)
            n_est = round(mean / p_est)
            p_est = mean / n_est

            self.n = int(n_est)
            self.p = float(p_est)

    def pmf(self, k):
        """Calculates the PMF for a given number of successes k."""
        if not isinstance(k, int):
            k = int(k)

        if k < 0 or k > self.n:
            return 0

        # factorial helper
        def factorial(x):
            result = 1
            for i in range(1, x + 1):
                result *= i
            return result

        # n choose k
        nck = factorial(self.n) / (factorial(k) *
                                   factorial(self.n - k))

        p = self.p
        q = 1 - p

        return nck * (p ** k) * (q ** (self.n - k))
   
 def pmf(self, k):
        """Calculates the PMF for a given number of successes k."""
        if not isinstance(k, int):
            k = int(k)

        if k < 0 or k > self.n:
            return 0

        # factorial helper
        def factorial(x):
            result = 1
            for i in range(1, x + 1):
                result *= i
            return result

        # n choose k
        nck = factorial(self.n) / (factorial(k) *
                                   factorial(self.n - k))

        p = self.p
        q = 1 - p

        return nck * (p ** k) * (q ** (self.n - k))
