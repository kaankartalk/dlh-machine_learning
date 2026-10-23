#!/usr/bin/env python3
"""LSTM cell"""
import numpy as np


class LSTMCell:
    """Represents an LSTM unit"""

    def __init__(self, i, h, o):
        """
        Class constructor

        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step

        h_prev: numpy.ndarray of shape (m, h), previous hidden state
        c_prev: numpy.ndarray of shape (m, h), previous cell state
        x_t: numpy.ndarray of shape (m, i), data input for the cell

        Returns: h_next, c_next, y
            h_next: next hidden state
            c_next: next cell state
            y: output of the cell (softmax)
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        f = self.sigmoid(np.matmul(concat, self.Wf) + self.bf)
        u = self.sigmoid(np.matmul(concat, self.Wu) + self.bu)
        c_tilde = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        o = self.sigmoid(np.matmul(concat, self.Wo) + self.bo)

        c_next = f * c_prev + u * c_tilde
        h_next = o * np.tanh(c_next)

        z = np.matmul(h_next, self.Wy) + self.by
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return h_next, c_next, y
