#!/usr/bin/env python3
"""Converts a label vector into a one-hot matrix."""
import tensorflow as tf


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix.

    Args:
        labels: label vector to convert
        classes: number of classes; if None, inferred from labels

    Returns:
        the one-hot matrix, with the last dimension being the
        number of classes
    """
    return tf.keras.utils.to_categorical(labels, num_classes=classes)
