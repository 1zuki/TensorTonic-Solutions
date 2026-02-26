import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len is None:
        max_len = 0
        
        for seg in seqs:
            max_len = max(len(seg), max_len)
            
    N = len(seqs)
    L = max_len

    arr = np.full((N, L), pad_value)
    
    for y in range(N):
        for x in range(min(L, len(seqs[y]))):
            arr[y][x] = seqs[y][x]
    
    return arr