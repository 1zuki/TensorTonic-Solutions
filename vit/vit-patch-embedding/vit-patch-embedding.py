import numpy as np
from sklearn.linear_model import LinearRegression  # for a trainable projection

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int) -> np.ndarray:
    """
    Convert image to patch embeddings.
    """
    B, H, W, C = image.shape
    P = patch_size
    
    # Calculate number of patches
    num_patches_h = H // P
    num_patches_w = W // P
    N = num_patches_h * num_patches_w
    
    # Reshape to separate patches: (B, H, W, C) -> (B, num_patches_h, P, num_patches_w, P, C)
    patches = image.reshape(B, num_patches_h, P, num_patches_w, P, C)
    
    # Permute axes: (B, num_patches_h, num_patches_w, P, P, C)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    
    # Flatten each patch: (B, N, P*P*C)
    patches_flat = patches.reshape(B, N, P * P * C)
    
    # Initialize projection matrix and bias
    # Shape: (P*P*C, D)
    weight = np.random.randn(P * P * C, embed_dim)
    bias = np.zeros(embed_dim)
    
    # Linear projection
    embeddings = patches_flat @ weight + bias
    
    return embeddings