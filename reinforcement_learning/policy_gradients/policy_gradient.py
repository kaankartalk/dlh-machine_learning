#!/usr/bin/env python3
"""Module that implements the policy function and Monte-Carlo policy gradient"""
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


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state and a weight
    matrix

    state: matrix representing the current observation of the environment
    weight: matrix of random weight

    Returns: the action and the gradient (in this order)
    """
    state = state.reshape(1, -1)
    probs = policy(state, weight)
    action = np.random.choice(probs.shape[1], p=probs[0])
    dsoftmax = probs.copy()
    dsoftmax[0, action] -= 1
    grad = -state.T @ dsoftmax
    return action, grad
