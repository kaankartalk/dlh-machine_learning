#!/usr/bin/env python3
"""Pooling Back Prop"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a
    neural network

    dA is a numpy.ndarray of shape (m, h_new, w_new, c_new)
        containing the partial derivatives with respect to the
        output of the pooling layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c is the number of channels
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c)
        containing the output of the previous layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
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

    Returns: the partial derivatives with respect to the previous
        layer (dA_prev)
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        a_prev = A_prev[i]

        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == 'max':
                        a_slice = a_prev[vert_start:vert_end,
                                          horiz_start:horiz_end, ch]
                        mask = (a_slice == np.max(a_slice))
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, ch] += \
                            mask * dA[i, h, w, ch]

                    elif mode == 'avg':
                        da = dA[i, h, w, ch]
                        average = da / (kh * kw)
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, ch] += \
                            np.ones((kh, kw)) * average

    return dA_prev
