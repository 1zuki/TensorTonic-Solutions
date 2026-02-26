import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y = np.array(y)
    if y.size == 0:
        return 0.0
        
    unique, count = np.unique(y, return_counts = "True")
    p = count / count.sum()
    p = p[p > 0]
    
    return np.sum(-p * np.log2(p))