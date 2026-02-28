import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    A = np.array(A)
    
    n = A.shape[0]
    idx = np.arange(n)
    
    return A[idx, idx].sum()