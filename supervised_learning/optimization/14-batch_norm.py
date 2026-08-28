#!/usr/bin/env python3
"""Creates a batch normalization layer for a neural network in
TensorFlow."""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in
    TensorFlow.

    Args:
        prev: the activated output of the previous layer
        n: the number of nodes in the layer to be created
        activation: the activation function that should be used on the
            output of the layer

    Returns:
        A tensor of the activated output for the layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n, kernel_initializer=initializer,
                                  use_bias=False)
    Z = layer(prev)

    mean, variance = tf.nn.moments(Z, axes=0)

    gamma = tf.Variable(tf.ones([1, n]), trainable=True, name='gamma')
    beta = tf.Variable(tf.zeros([1, n]), trainable=True, name='beta')

    Z_norm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma, 1e-7)

    return activation(Z_norm)
