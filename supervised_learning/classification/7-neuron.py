#!/usr/bin/env python3
"""Defines a single neuron performing binary classification."""
import matplotlib.pyplot as plt
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

    def forward_prop(self, X):
        """Calculate the forward propagation of the neuron.

        Args:
            X (numpy.ndarray): input data of shape (nx, m), where nx is the
                number of input features and m is the number of examples.

        Returns:
            numpy.ndarray: the activated output __A.
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

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
        """Evaluate the neuron's predictions.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).
            Y (numpy.ndarray): correct labels of shape (1, m).

        Returns:
            tuple: the predicted labels (1 if output >= 0.5, else 0) and the
                cost of the network.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculate one pass of gradient descent on the neuron.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).
            Y (numpy.ndarray): correct labels of shape (1, m).
            A (numpy.ndarray): activated output of shape (1, m).
            alpha (float): the learning rate.
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = np.matmul(dZ, X.T) / m
        db = np.sum(dZ) / m
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the neuron.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).
            Y (numpy.ndarray): correct labels of shape (1, m).
            iterations (int): number of iterations to train over.
            alpha (float): the learning rate.
            verbose (bool): whether to print the cost every step iterations.
            graph (bool): whether to plot the cost once training completes.
            step (int): interval at which cost is printed and plotted.

        Raises:
            TypeError: if iterations is not an integer, alpha is not a float,
                or step is not an integer.
            ValueError: if iterations, alpha, or step is not positive, or if
                step is greater than iterations.

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
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step <= 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        costs = []
        steps = []

        for i in range(iterations + 1):
            A = self.forward_prop(X)
            if (verbose or graph) and (i % step == 0 or i == iterations):
                cost = self.cost(Y, A)
                costs.append(cost)
                steps.append(i)
                if verbose:
                    print('Cost after {} iterations: {}'.format(i, cost))
            if i < iterations:
                self.gradient_descent(X, Y, A, alpha)

        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
