import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    x = np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)

    if x.ndim == 2:
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)

        x_hat = (x - mean) / np.sqrt(var + eps)
        y = gamma * x_hat + beta

    elif x.ndim == 4:
        mean = np.mean(x, axis=(0,2,3), keepdims=True)
        var = np.var(x, axis=(0,2,3), keepdims=True)

        x_hat = (x - mean) / np.sqrt(var + eps)

        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)

        y = gamma * x_hat + beta

    else:
        raise ValueError("Input must be (N,D) or (N,C,H,W)")

    return y