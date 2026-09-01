import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    n = x.shape[0]

    variance = np.sum((x - np.mean(x)) ** 2) / (n - 1)
    deviation = np.sqrt(variance)

    return {"variance": float(variance), "standard_deviation": float(deviation)}