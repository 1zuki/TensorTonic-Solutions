import numpy as np

def robust_scaling(values):
    arr = np.asarray(values)
    sorted_vals = np.sort(arr)
    n = len(arr)

    if n == 1:
        return [0]
    
    if n % 2 == 1:
        median = sorted_vals[n // 2]
        lower = sorted_vals[:n // 2]
        upper = sorted_vals[n // 2 + 1:]
    else:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        lower = sorted_vals[:n // 2]
        upper = sorted_vals[n // 2:]

    m = len(lower)
    q1 = lower[m // 2] if m % 2 else (lower[m // 2 - 1] + lower[m // 2]) / 2
    
    m = len(upper)
    q3 = upper[m // 2] if m % 2 else (upper[m // 2 - 1] + upper[m // 2]) / 2
    
    iqr = q3 - q1
    
    if iqr == 0:
        return arr - median
    
    return (arr - median) / iqr