#!/usr/bin/env python3
"""Contrast"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image

    image is a 3D tf.Tensor representing the input image to
        adjust the contrast
    lower is a float representing the lower bound of the random
        contrast factor range
    upper is a float representing the upper bound of the random
        contrast factor range

    Returns the contrast-adjusted image
    """
    return tf.image.random_contrast(image, lower, upper)
