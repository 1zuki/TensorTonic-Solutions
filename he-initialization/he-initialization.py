def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    # Write code here
    W = np.asarray(W)
    fan_in = np.asarray(fan_in)

    
    limit = np.sqrt(6 / fan_in)
    new_W = W * (2 * limit) - limit

    return new_W