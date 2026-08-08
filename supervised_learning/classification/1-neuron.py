#!/usr/bin/env python3
"""Defines a single neuron performing binary classification."""
import numpy as np


class Neuron:
    """A single neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize the neuron.

        Args:
            nx (int): number of input features to the neuron.

        Raises:
            TypeError: if nx is not an integer.
            ValueError: if nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be a integer')
        if nx < 1:
            raise ValueError('nx must be positive')

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """numpy.ndarray: the weights vector for the neuron."""
        return self.__W

    @property
    def b(self):
        """int: the bias for the neuron."""
        return self.__b

    @property
    def A(self):
        """int: the activated output of the neuron."""
        return self.__A
