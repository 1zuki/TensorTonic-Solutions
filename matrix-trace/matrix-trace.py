import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    A = np.array(A)
    sum = 0

    idx = np.arange(A.shape[0])
    sum += A[idx, idx].sum()

    return sum