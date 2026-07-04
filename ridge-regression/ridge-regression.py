import numpy as np

def ridge_regression(X, y, lam):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # If X is 1D, treat it as one feature column
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    d = X.shape[1]
    I = np.eye(d)

    return np.linalg.solve(X.T @ X + lam * I, X.T @ y)