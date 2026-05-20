#!/usr/bin/env python3
"""
Determinant of a matrix
"""


def determinant(matrix):
    """Calculates the determinant of a matrix"""

    # --- Type checks ---
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if matrix == []:
        raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    # --- 0x0 matrix case ---
    if matrix == [[]]:
        return 1

    # --- Square check ---
    size = len(matrix)
    for row in matrix:
        if len(row) != size:
            raise ValueError("matrix must be a square matrix")

    # --- 1x1 ---
    if size == 1:
        return matrix[0][0]

    # --- 2x2 ---
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # --- Recursive expansion ---
    det = 0
    for col in range(size):
        # Build minor
        minor = []
        for r in range(1, size):
            row = []
            for c in range(size):
                if c != col:
                    row.append(matrix[r][c])
            minor.append(row)

        det += ((-1) ** col) * matrix[0][col] * determinant(minor)

    return det
