import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    # YOUR CODE HERE
    grad_norms = []

    norm = np.linalg.norm(W_hh, ord = 2)
    grad = 1.0

    for t in range(T):
        grad *= norm
        grad_norms.append(grad)

    return grad_norms
