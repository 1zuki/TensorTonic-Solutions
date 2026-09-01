import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    # Write code here
    x = np.asarray(x, dtype=float)

    pmf = np.where(x == 0, 1 - p, p)
    variance = p * (1 - p)

    return {"pmf": pmf, "mean": float(p), "variance": float(variance)}