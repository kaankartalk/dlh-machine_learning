#!/usr/bin/env python3
"""
Module for transposing a 2D matrix.
"""


df matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix.
    """
    transposed = []
    for col in range(len(matrix[0])):
        new_row = []
        for row in range(len(matrix)):
            new_row.append(matrix[row][col])
        transposed.append(new_row)
    return transposed
