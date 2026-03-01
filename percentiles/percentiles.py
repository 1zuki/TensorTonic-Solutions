import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    x = np.sort(np.array(x))
    q = np.array(q) 

    percentiles = []
    for p in q:
        L = (p / 100) * (x.size - 1)

        lower = int(np.floor(L))
        upper = int(np.ceil(L))

        if lower == upper:
            percentiles.append(float(x[lower]))
            
        else:
            temp = x[lower] + (L - lower) * (x[upper] - x[lower])
            percentiles.append(float(temp))

    return np.array(percentiles)