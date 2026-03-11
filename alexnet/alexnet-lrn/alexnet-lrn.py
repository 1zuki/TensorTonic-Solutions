import numpy as np

def local_response_normalization(x: np.ndarray, k: float = 2, n: int = 5,
                                  alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """Apply Local Response Normalization across channels."""
    # YOUR CODE HERE
    batch, H, W, C = x.shape
    half = n // 2;
    
    x_sq = x ** 2
    out = np.zeros(x.shape)

    for c in range(C):
        start = max(0, c - half)
        end = min(C - 1, c + half)

        sq_sum = np.sum(x_sq[:, :, :, start : end], axis = 3)
        norm = (k + alpha * sq_sum) ** beta
        out[:, :, :, c] = x[:, :, :, c] / norm

    return out