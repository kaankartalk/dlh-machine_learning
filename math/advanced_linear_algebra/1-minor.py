#!/usr/bin/env python3
"""
Minor matrix
"""

determinant = __import__('0-determinant').determinant


def minor(matrix):
    """Calculates the minor matrix of a matrix"""

    # --- Type checks ---
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if matrix == []:
        raise ValueError("matrix must be a non-empty square matrix")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    # --- Square check ---
    size = len(matrix)
    for row in matrix:
        if len(row) != size:
            raise ValueError("matrix must be a non-empty square matrix")

    # --- 1x1 matrix ---
    if size == 1:
        return [[1]]

    # --- Build minor matrix ---
    minor_matrix = []

    for i in range(size):
        row_minors = []
        for j in range(size):
            # Build submatrix by removing row i and column j
            sub = []
            for r in range(size):
                if r != i:
                    new_row = []
                    for c in range(size):
                        if c != j:
                            new_row.append(matrix[r][c])
                    sub.append(new_row)

            # determinant of submatrix
            row_minors.append(determinant(sub))

        minor_matrix.append(row_minors)

    return minor_matrix
