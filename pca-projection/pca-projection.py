import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    X = np.asarray(X)
    n, d = X.shape

    mean = np.mean(X, axis = 0)
    X_c = X - mean

    covariance = (X_c.T @ X_c) / (n - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    idx = np.argsort(eigenvalues)[::-1]

    W = eigenvectors[:, idx[:k]]

    X_proj = X_c @ W

    return X_proj