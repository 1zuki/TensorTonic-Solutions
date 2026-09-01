import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    
    numerator = np.dot(a, b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    return 0.0 if denominator == 0 else float(numerator / denominator)