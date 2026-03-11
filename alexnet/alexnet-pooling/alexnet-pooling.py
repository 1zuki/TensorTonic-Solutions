import numpy as np

def max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """Apply 2D max pooling (shape simulation)."""
    # YOUR CODE HERE
    x = np.asarray(x)
    batch, H, W, C = x.shape

    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1

    return np.zeros((batch, H_out, W_out, C))