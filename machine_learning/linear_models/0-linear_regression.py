#!/usr/bin/env python3
"""Defines a function that builds an untrained linear regression model."""
from sklearn import linear_model


def Linear_Regression():
    """Create an untrained ordinary least squares Linear Regression model.

    Returns:
        model: An untrained sklearn.linear_model.LinearRegression instance.
    """
    model = linear_model.LinearRegression()
    return model