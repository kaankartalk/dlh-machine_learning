#!/usr/bin/env python3
"""
Adds two 2D matrices element-wise.
"""


def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices element-wise."""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    new_matrix = []
    for row1, row2 in zip(mat1, mat2):
        new_matrix.append([a + b for a, b in zip(row1, row2)])

    return new_matrix
