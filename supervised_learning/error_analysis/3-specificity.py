#!/usr/bin/env python3
"""Calculates the specificity for each class in a confusion matrix."""
import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class in a confusion
    matrix.

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column
            indices represent the predicted labels

    Returns:
        a numpy.ndarray of shape (classes,) containing the specificity
        of each class
    """
    total = np.sum(confusion)
    true_positives = np.diagonal(confusion)
    actual_positives = np.sum(confusion, axis=1)
    predicted_positives = np.sum(confusion, axis=0)

    false_positives = predicted_positives - true_positives
    actual_negatives = total - actual_positives
    true_negatives = actual_negatives - false_positives

    return true_negatives / actual_negatives
