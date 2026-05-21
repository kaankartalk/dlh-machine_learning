#!/usr/bin/env python3
"""
Module for calculating the cofactor matrix of a square matrix.
"""


def cofactor(matrix):
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    def det(m):
        if m == [] or m == [[]]:
            return 1
        if len(m) == 1:
            return m[0][0]
        if len(m) == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]

        total = 0
        for c in range(len(m)):
            minor = []
            for r in range(1, len(m)):
                row = m[r][:c] + m[r][c + 1:]
                minor.append(row)
            total += ((-1) ** c) * m[0][c] * det(minor)
        return total

    cof = []
    for i in range(n):
        row_cof = []
        for j in range(n):
            minor = []
            for r in range(n):
                if r != i:
                    row = matrix[r][:j] + matrix[r][j + 1:]
                    minor.append(row)
            value = ((-1) ** (i + j)) * det(minor)
            row_cof.append(value)
        cof.append(row_cof)

    return cof
