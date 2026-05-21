#!/usr/bin/env python3

def cofactor(matrix):
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError()
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError()

    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError()

    def determinant(mat):
        if mat == [] or mat == [[]]:
            return 1
        if len(mat) == 1:
            return mat[0][0]
        if len(mat) == 2:
            return mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0]

        det = 0
        for c in range(len(mat)):
            minor = []
            for r in range(1, len(mat)):
                row = []
                for k in range(len(mat)):
                    if k != c:
                        row.append(mat[r][k])
                minor.append(row)
            det += ((-1)**c) * mat[0][c] * determinant(minor)
        return det

    cof = []
    for i in range(n):
        row_cof = []
        for j in range(n):
            minor = []
            for r in range(n):
                if r != i:
