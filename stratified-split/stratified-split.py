import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    # Write code here
    X = np.asarray(X)
    y = np.asarray(y)

    train_idx = []
    test_idx = []

    for cls_cnt in np.unique(y):
        cls_idx = np.where(y == cls_cnt)[0].copy()

        if rng is not None:
            rng.shuffle(cls_idx)
        else:
            np.random.shuggle(cls_idx)

        n_cls = len(cls_idx)
        n_test = int(round(n_cls * test_size))

        if n_cls > 1:
            n_test = min(n_test, n_cls - 1)

        test_idx.append(cls_idx[:n_test])
        train_idx.append(cls_idx[n_test:])

    train_idx = np.sort(np.concatenate(train_idx).astype(int))
    test_idx = np.sort(np.concatenate(test_idx).astype(int))

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]