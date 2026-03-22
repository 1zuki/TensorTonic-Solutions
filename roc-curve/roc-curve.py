import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # sort by descending score
    order = np.argsort(-y_score, kind = "mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    # cumulative TP/FP
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    # keep only last index of each score group
    distinct = np.where(np.diff(y_score_sorted) != 0)[0]
    threshold_idxs = np.r_[distinct, len(y_score_sorted) - 1]

    tps = tps[threshold_idxs]
    fps = fps[threshold_idxs]
    thresholds = y_score_sorted[threshold_idxs]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    tpr = tps / P
    fpr = fps / N

    # add starting point
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[np.inf, thresholds]

    return fpr, tpr, thresholds