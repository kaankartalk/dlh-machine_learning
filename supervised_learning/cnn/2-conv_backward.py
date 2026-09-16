#!/usr/bin/env python3
"""Convolutional Back Prop"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer
    of a neural network

    dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new)
        containing the partial derivatives with respect to the
        unactivated output of the convolutional layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c_new is the number of channels in the output
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
        the kernels for the convolution
        kh is the filter height
        kw is the filter width
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the
        biases applied to the convolution
    padding is a string that is either same or valid, indicating
        the type of padding used
    stride is a tuple of (sh, sw) containing the strides for the
        convolution
        sh is the stride for the height
        sw is the stride for the width

    Returns: the partial derivatives with respect to the previous
        layer (dA_prev), the kernels (dW), and the biases (db),
        respectively
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2) + 1
    else:
        ph, pw = 0, 0

    A_prev_pad = np.pad(A_prev,
                         ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                         mode='constant', constant_values=0)

    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice = a_prev_pad[vert_start:vert_end,
                                          horiz_start:horiz_end, :]

                    da_prev_pad[vert_start:vert_end,
                                horiz_start:horiz_end, :] += \
                        W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

        if padding == 'same':
            dA_prev_pad_i = da_prev_pad[ph:-ph, pw:-pw, :]
        else:
            dA_prev_pad_i = da_prev_pad

        dA_prev_pad[i] = da_prev_pad
        if i == 0:
            dA_prev = np.zeros((m, ) + dA_prev_pad_i.shape)
        dA_prev[i] = dA_prev_pad_i

    return dA_prev, dW, db
