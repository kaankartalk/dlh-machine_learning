#!/usr/bin/env python3
"""Defines a neural network with one hidden layer."""
import numpy as np


class NeuralNetwork:
    """A neural network with one hidden layer performing binary
    classification."""

    def __init__(self, nx, nodes):
        """Initialize the neural network.

        Args:
            nx (int): number of input features.
            nodes (int): number of nodes in the hidden layer.

        Raises:
            TypeError: if nx or nodes is not an integer.
            ValueError: if nx or nodes is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """numpy.ndarray: the weights vector for the hidden layer."""
        return self.__W1

    @property
    def b1(self):
        """numpy.ndarray: the bias for the hidden layer."""
        return self.__b1

    @property
    def A1(self):
        """int: the activated output for the hidden layer."""
        return self.__A1

    @property
    def W2(self):
        """numpy.ndarray: the weights vector for the output neuron."""
        return self.__W2

    @property
    def b2(self):
        """int: the bias for the output neuron."""
        return self.__b2

    @property
    def A2(self):
        """int: the activated output for the output neuron."""
        return self.__A2

    def forward_prop(self, X):
        """Calculate the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): input data of shape (nx, m), where nx is the
                number of input features and m is the number of examples.

        Returns:
            tuple: the activated outputs __A1 and __A2, respectively.
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2
