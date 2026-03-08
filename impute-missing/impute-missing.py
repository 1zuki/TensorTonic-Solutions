import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    # Write code here
    x = np.array(X, dtype=float).copy()

    if x.ndim == 1:
        valid = ~np.isnan(x)

        if not np.any(valid):
            stat = 0.0
        else:
            if strategy == 'mean':
                stat = np.mean(x[valid])
            else:
                stat = np.median(x[valid])

        x[np.isnan(x)] = stat
        return x

    for j in range(x.shape[1]):

        col = x[:, j]
        valid = ~np.isnan(col)

        if not np.any(valid):
            stat = 0.0
        else:
            if strategy == 'mean':
                stat = np.mean(col[valid])
            else:
                stat = np.median(col[valid])

        col[np.isnan(col)] = stat
        x[:, j] = col

    return x