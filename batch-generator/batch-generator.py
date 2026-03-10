import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    # Write code here
    X = np.asarray(X)
    y = np.asarray(y)
    N = len(X)
    indices = np.arange(N)
    
    if rng is not None:
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)

    if drop_last:
        N -= batch_size
    
    for start in range(0, N, batch_size):
        end = start + batch_size
        X_batch = X[indices[start:end]]
        Y_batch = y[indices[start:end]]
        yield X_batch, Y_batch