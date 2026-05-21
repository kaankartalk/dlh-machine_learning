#!/usr/bin/env python3

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
            return m[0][0]*m[1][1] - m[0][1]*m[1][0]

        s = 0
        for c in range(len(m)):
            minor = [row[:c] + row[c+1:] for row in m[1:]]
            s += ((-1)**c) * m[0][c] * det(minor)
        return s

    return [
        [
            ((-1)**(i+j)) * det([row[:j] + row[j+1:] for r, row in enumerate(matrix) if r != i])
            for j in range(n)
        ]
        for i in range(n)
    ]
