#!/usr/bin/env python3
"""Scaled dot product attention"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention

    Q: tensor with last two dimensions (..., seq_len_q, dk), the queries
    K: tensor with last two dimensions (..., seq_len_v, dk), the keys
    V: tensor with last two dimensions (..., seq_len_v, dv), the values
    mask: optional tensor that can be broadcast into
          (..., seq_len_q, seq_len_v), or None

    Returns: output, weights
        output: tensor of shape (..., seq_len_q, dv) with the attention
        weights: tensor of shape (..., seq_len_q, seq_len_v) with the
                 attention weights
    """
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(dk)

    if mask is not None:
        scaled += mask * -1e9

    weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights
