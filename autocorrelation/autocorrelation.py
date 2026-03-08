import numpy as np

def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    # Write code here
    x = np.mean(series)
    gamma = np.sum((series - x) ** 2)

    if gamma == 0:
        return [1] + [0] * max_lag
    
    r = []
    for k in range(max_lag + 1):
        sum = 0
        for i in range(len(series) - k):
            sum += (series[i] - x) * (series[i + k] - x)

        r.append(float(sum / gamma))

    return r
        