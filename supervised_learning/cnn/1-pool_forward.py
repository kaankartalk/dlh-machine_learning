#!/usr/bin/env python3
"""Pooling Forward Prop"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer
    of a neural network

    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the
        kernel for the pooling
        kh is the kernel height
        kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the
        pooling
        sh is the stride for the height
        sw is the stride for the width
    mode is a string containing either max or avg, indicating
        whether to perform maximum or average pooling, respectively

    Returns: the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = int((h_prev - kh) / sh) + 1
    w_new = int((w_prev - kw) / sw) + 1

    A = np.zeros((m, h_new, w_new, c_prev))

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            A_slice = A_prev[:, vert_start:vert_end,
                              horiz_start:horiz_end, :]

            if mode == 'max':
                A[:, i, j, :] = np.max(A_slice, axis=(1, 2))
            elif mode == 'avg':
                A[:, i, j, :] = np.mean(A_slice, axis=(1, 2))

    return A
