#!/usr/bin/env python3
"""Module that implements the policy function for policy gradients"""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix

    matrix: numpy.ndarray of shape (n, d) containing the states
    weight: numpy.ndarray of shape (d, a) containing the weights

    Returns: numpy.ndarray of shape (n, a) with the action probabilities
    """
    z = matrix @ weight
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)
