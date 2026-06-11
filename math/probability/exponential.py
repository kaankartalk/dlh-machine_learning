#!/usr/bin/env python3
"""Exponential distribution class"""


class Exponential:
    """Represents an exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Class constructor"""

        if data is None:
            # lambtha must be positive
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

        else:
            # data must be a list
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            # must contain at least 2 points
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # exponential lambtha = 1 / mean
            mean = sum(data) / len(data)
            self.lambtha = 1 / mean

    def pdf(self, x):
        """Calculates the PDF for a given time period x"""

        if x < 0:
            return 0

        # required e value from project
        e = 2.7182818285
        lambtha = self.lambtha

        return lambtha * (e ** (-lambtha * x))
