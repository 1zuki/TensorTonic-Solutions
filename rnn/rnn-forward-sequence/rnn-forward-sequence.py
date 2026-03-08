import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE
    batch, time, dim = X.shape
    hidden = []
    h = h_0
    
    for t in range(time):
        x_t = X[:, t, :]
        
        h = np.tanh(x_t @ W_xh.T + h @ W_hh.T + b_h)
        
        hidden.append(h)

    h_all = np.stack(hidden, axis=1)
    h_final = h

    return h_all, h_final