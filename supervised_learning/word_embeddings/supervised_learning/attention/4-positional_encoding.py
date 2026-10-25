#!/usr/bin/env python3
"""Positional encoding for a transformer"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer

    max_seq_len: integer, the maximum sequence length
    dm: the model depth

    Returns: numpy.ndarray of shape (max_seq_len, dm) containing the
             positional encoding vectors
    """
    position = np.arange(max_seq_len)[:, np.newaxis]
    even = np.arange(0, dm, 2)[np.newaxis, :]
    angles = position / np.power(10000, even / dm)

    PE = np.zeros((max_seq_len, dm))
    PE[:, 0::2] = np.sin(angles)
    PE[:, 1::2] = np.cos(angles[:, :dm // 2])

    return PE
