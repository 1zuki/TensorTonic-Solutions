import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.array(X)
    
    u = np.mean(X, axis = 0)
    X_cen = X - u

    N = len(X)
    if X.ndim != 2 or N < 2:
        return None

    sigma = (1 / (N - 1)) * X_cen.T @ X_cen
    return sigma