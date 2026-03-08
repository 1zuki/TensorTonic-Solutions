import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here

    if len(x.shape) == 3:        # (C,H,W)
        return np.mean(x, axis=(1,2))

    elif len(x.shape) == 4:      # (N,C,H,W)
        return np.mean(x, axis=(2,3))

    else:
        raise ValueError    

    