def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    ELU = []
    
    for i in range(len(x)):
        if x[i] > 0:
            ELU.append(x[i])
        else:
            ELU.append(alpha * (math.exp(x[i]) - 1))
    return ELU