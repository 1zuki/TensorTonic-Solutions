import numpy as np

def unet_encoder_block(x: np.ndarray, out_channels: int) -> tuple:
    """
    U-Net encoder block: double conv + max pool.
    """
    # Your implementation here
    batch, H, W, C = x.shape

    H_out = H - 4
    W_out = W - 4

    H_pool = H_out // 2
    W_pool = W_out // 2

    return np.zeros((batch, H_pool, W_pool, out_channels)), np.zeros((batch, H_out, W_out, out_channels))