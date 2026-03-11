import numpy as np

def random_crop(image: np.ndarray, crop_size: int = 224) -> np.ndarray:
    """Extract a random crop from the image."""
    # YOUR CODE HERE
    H, W, _ = image.shape
    
    max_x = H - crop_size
    max_y = W - crop_size
    
    start_x = np.random.randint(0, max_x + 1)
    start_y = np.random.randint(0, max_y + 1)
    
    return image[start_x:start_x+crop_size, start_y:start_y+crop_size, :]

def random_horizontal_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Randomly flip image horizontally."""
    # YOUR CODE HERE
    if np.random.rand() < p:
        return np.fliplr(image)

    return image

