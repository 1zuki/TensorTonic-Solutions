import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    sig = np.array(x)
    return (1 / (1 + np.exp(-sig)))