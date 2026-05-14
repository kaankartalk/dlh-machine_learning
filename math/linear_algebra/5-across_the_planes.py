def add_matrices2D(mat1, mat2):
    # Check if shapes match
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    # Create new matrix with element-wise sum
    new_matrix = []
    for row1, row2 in zip(mat1, mat2):
        new_matrix.append([a + b for a, b in zip(row1, row2)])

    return new_matrix
