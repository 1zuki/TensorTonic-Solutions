import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation)."""

    batch, H, W, C = image.shape

    kernel = 11
    stride = 4
    filters = 96

    H_out = (H - kernel + H % kernel) // stride + 1
    W_out = (W - kernel + H % kernel) // stride + 1

    return np.zeros((batch, H_out, W_out, filters))