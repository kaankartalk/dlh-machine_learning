#!/usr/bin/env python3

def cofactor(matrix):
    # --- type checks ---
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    # --- square & non-empty check ---
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # --- helper: determinant ---
    def determinant(mat):
        if mat == [[]]:
            return 1
        n = len(mat)
        if n == 1:
            return mat[0][0]
        if n == 2:
            return mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0]

        det = 0
        for col in range(n):
            minor = []
            for r in range(1, n):
                row = []
                for c in range(n):
                    if c != col:
                        row.append(mat[r][c])
                minor.append(row)
            det += ((-1) ** col) * mat[0][col] * determinant(minor)
        return det

    # --- compute cofactor matrix ---
    cof = []
    for i in range(size):
        cof_row = []
        for j in range(size):
            # build minor for element (i, j)
            minor = []
            for r in range(size):
                if r != i:
                    row = []
                    for c in range(size):
                        if c != j:
                            row.append(matrix[r][c])
                    minor.append(row)

            sign = (-1) ** (i + j)
            cof_row.append(sign * determinant(minor))
        cof.append(cof_row)

    return cof
