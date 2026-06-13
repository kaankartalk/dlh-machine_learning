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

            # Variance
            var = sum((x - mean) ** 2 for x in data) / len(data)

            # Method of Moments:
            # p = 1 - (var / mean)
            p_est = 1 - (var / mean)

            # n = mean / p
            n_est = round(mean / p_est)

            # Recalculate p using rounded n
            p_est = mean / n_est

            self.n = int(n_est)
            self.p = float(p_est)
