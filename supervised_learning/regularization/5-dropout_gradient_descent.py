#!/usr/bin/env python3
"""Updates the weights of a neural network with Dropout regularization
using gradient descent.
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights of a neural network with Dropout
    regularization using gradient descent.

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) that contains
            the correct labels for the data
        weights: dictionary of the weights and biases of the neural
            network
        cache: dictionary of the outputs and dropout masks of each
            layer of the neural network
        alpha: the learning rate
        keep_prob: the probability that a node will be kept
        L: the number of layers of the network

    Returns:
        None; the weights of the network are updated in place
    """
    m = Y.shape[1]
    weights_copy = {key: value.copy() for key, value in weights.items()}
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights_copy['W' + str(i)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dA = np.matmul(W.T, dZ)
            dA = dA * cache['D' + str(i - 1)]
            dA = dA / keep_prob
            dZ = dA * (1 - A_prev ** 2)

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
