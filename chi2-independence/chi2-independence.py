import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    # Write code here
    C = np.array(C)

    row_sum = np.sum(C, axis=1, keepdims=True)
    col_sum = np.sum(C, axis=0, keepdims=True)
    total = np.sum(C)

    expected = row_sum @ col_sum / total

    chi2 = np.sum((C - expected) ** 2 / expected)

    return chi2, expected