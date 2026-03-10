import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x)

    was_1d = False
    if x.ndim == 1:
        x = x.reshape(1, -1)
        was_1d = True

    dx, dy = x.shape

    if rng is None:
        r = np.random.random((dx, dy))
    else:
        r = rng.random((dx, dy))

    pattern = (r < (1 - p)) / (1 - p)
    output = x * pattern

    if was_1d:
        output = output.reshape(-1)
        pattern = pattern.reshape(-1)

    return output, pattern