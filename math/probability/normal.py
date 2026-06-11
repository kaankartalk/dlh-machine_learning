#!/usr/bin/env python3
"""Normal distribution class"""


class Normal:
    """Represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Class constructor"""

        if data is None:
            # stddev must be positive
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")

            self.mean = float(mean)
            self.stddev = float(stddev)

        else:
            # data must be a list
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            # must contain at least 2 points
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # calculate mean
            self.mean = sum(data) / len(data)

            # calculate standard deviation
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return (z * self.stddev) + self.mean
    def pdf(self, x):
        """Calculates the PDF for a given x-value"""

        e = 2.7182818285
        pi = 3.1415926536

        mean = self.mean
        stddev = self.stddev

        # coefficient: 1 / (stddev * sqrt(2π))
        coeff = 1 / (stddev * ((2 * pi) ** 0.5))

        # exponent: -0.5 * ((x - mean) / stddev)^2
        exponent = -0.5 * (((x - mean) / stddev) ** 2)

        return coeff * (e ** exponent)
