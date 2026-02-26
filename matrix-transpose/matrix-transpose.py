import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.array(A)
    AT = np.zeros(A.shape[::-1])

    for y in range(len(AT)):
        for x in range(len(AT[y])):
            AT[y][x] = A[x][y]

    return AT