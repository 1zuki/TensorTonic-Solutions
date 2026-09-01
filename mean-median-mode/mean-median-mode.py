from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    # Write code here
    x = np.asarray(x, dtype=float)

    vals, cnts = np.unique(x, return_counts=True)
    max_cnt = np.max(cnts)
    best_vals = vals[np.argwhere(cnts == max_cnt).flatten()]
    
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "mode": float(best_vals[0])}