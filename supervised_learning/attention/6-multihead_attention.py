#!/usr/bin/env python3
"""Multi head attention"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Performs multi head attention"""

    def __init__(self, dm, h):
        """
        Class constructor

        dm: integer, the dimensionality of the model
        h: integer, the number of heads
           dm is divisible by h

        Sets the public instance attributes:
            h: the number of heads
            dm: the dimensionality of the model
            depth: the depth of each attention head
            Wq: Dense layer with dm units for the query matrix
            Wk: Dense layer with dm units for the key matrix
            Wv: Dense layer with dm units for the value matrix
            linear: Dense layer with dm units for the attention output
        """
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch):
        """
        Splits the last dimension into (h, depth) and moves the heads
        dimension before the sequence dimension

        x: tensor of shape (batch, seq_len, dm)
        batch: the batch size

        Returns: tensor of shape (batch, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Calculates the multi head attention

        Q: tensor of shape (batch, seq_len_q, dk), input for the queries
        K: tensor of shape (batch, seq_len_v, dk), input for the keys
        V: tensor of shape (batch, seq_len_v, dv), input for the values
        mask: always None

        Returns: output, weights
            output: tensor of shape (..., seq_len_q, dm) with the
                    attention output
            weights: tensor of shape (..., h, seq_len_q, seq_len_v) with
                     the attention weights
        """
        batch = tf.shape(Q)[0]

        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        q = self.split_heads(q, batch)
        k = self.split_heads(k, batch)
        v = self.split_heads(v, batch)

        attention, weights = sdp_attention(q, k, v, mask)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat = tf.reshape(attention, (batch, -1, self.dm))

        output = self.linear(concat)

        return output, weights
