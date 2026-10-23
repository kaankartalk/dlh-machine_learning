#!/usr/bin/env python3
"""Forward propagation for a deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN

    rnn_cells: list of RNNCell instances of length l
        l: number of layers
    X: numpy.ndarray of shape (t, m, i), the input data
        t: maximum number of time steps
        m: batch size
        i: dimensionality of the data
    h_0: numpy.ndarray of shape (l, m, h), the initial hidden state
        h: dimensionality of the hidden state

    Returns: H, Y
        H: numpy.ndarray containing all of the hidden states
        Y: numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].by.shape[1]

    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0

    for step in range(t):
        x = X[step]
        for layer in range(l):
            h_next, y = rnn_cells[layer].forward(H[step, layer], x)
            H[step + 1, layer] = h_next
            x = h_next
        Y[step] = y

    return H, Y
