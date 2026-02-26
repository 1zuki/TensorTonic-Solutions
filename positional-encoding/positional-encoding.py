import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    PE = np.empty((seq_len, d_model))

    for y in range(seq_len):
        for x in range(d_model):
            i = x // 2
            
            if x % 2 == 0:
                PE[y, x] = np.sin(y / base ** ((2 * i) / d_model))
            else:
                PE[y, x] = np.cos(y / base ** ((2 * i) / d_model))
                
    return PE