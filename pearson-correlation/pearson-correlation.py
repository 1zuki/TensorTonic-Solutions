import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    # Write code here
    X = np.asarray(X, dtype=float)

    centered = X - np.mean(X, axis=0)
    covar = centered.T @ centered / (X.shape[0] - 1)
    
    standard_dev = np.sqrt(np.diag(covar))
    denorm = np.outer(standard_dev, standard_dev)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        return covar / denorm