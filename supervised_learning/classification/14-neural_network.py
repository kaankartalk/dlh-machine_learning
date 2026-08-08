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

    def cost(self, Y, A):
        """Calculate the cost of the model using logistic regression.

        Args:
            Y (numpy.ndarray): correct labels of shape (1, m).
            A (numpy.ndarray): activated output of shape (1, m).

        Returns:
            float: the logistic regression cost.
        """
        m = Y.shape[1]
        losses = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        return -np.sum(losses) / m

    def evaluate(self, X, Y):
        """Evaluate the neural network's predictions.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).
            Y (numpy.ndarray): correct labels of shape (1, m).

        Returns:
            tuple: the predicted labels (1 if output >= 0.5, else 0) and the
                cost of the network.
        """
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculate one pass of gradient descent on the neural network.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).
            Y (numpy.ndarray): correct labels of shape (1, m).
            A1 (numpy.ndarray): output of the hidden layer.
            A2 (numpy.ndarray): predicted output.
            alpha (float): the learning rate.
        """
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = np.matmul(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Train the neural network.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).
            Y (numpy.ndarray): correct labels of shape (1, m).
            iterations (int): number of iterations to train over.
            alpha (float): the learning rate.

        Raises:
            TypeError: if iterations is not an integer or alpha is not a float.
            ValueError: if iterations or alpha is not positive.

        Returns:
            tuple: the evaluation of the training data after training.
        """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')

        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)
