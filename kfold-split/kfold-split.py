import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    # Write code here
    N = np.array(N)
    indices = np.arange(N)

    if shuffle:
        if rng is None:
            np.random.shuffle(indices)
        else:
            indices = rng.permutation(indices)

    fold_sizes = [N // k] * k
    for i in range(N % k):
        fold_sizes[i] += 1

    splits = []
    start = 0

    for size in fold_sizes:
        end = start + size

        val_idx = indices[start:end]
        train_idx = np.concatenate((indices[:start], indices[end:]))

        splits.append((train_idx, val_idx))

        start = end

    return splits