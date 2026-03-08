import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    # Write code here
    x = np.array(x)
    
    s = np.sqrt(np.sum((x - np.mean(x)) ** 2 / (len(x) - 1)))
    t = (np.mean(x) - mu0) / (s / (len(x) ** (1 / 2)))

    return t