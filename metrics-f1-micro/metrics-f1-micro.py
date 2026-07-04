import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    TP = TN = FP = FN = 0

    for c in classes:
        TP += np.sum((y_true == c) & (y_pred == c))
        FP += np.sum((y_true != c) & (y_pred == c))
        TN += np.sum((y_true != c) & (y_pred != c))
        FN += np.sum((y_true == c) & (y_pred != c))

    return 2 * TP / (2 * TP + FP + FN)