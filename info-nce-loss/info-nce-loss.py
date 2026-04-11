import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    # Write code here
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)

    S = (Z1 @ Z2.T) / temperature

    S_max = np.max(S, axis=1, keepdims=True)
    S_stable = S - S_max

    log = np.log(np.sum(np.exp(S_stable), axis=1)) + S_max[:, 0]

    positive_pairs = np.diag(S)
    loss = -np.mean(positive_pairs - log)

    return float(loss)
    