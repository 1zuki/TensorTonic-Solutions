def _dot(a, b):
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))

def lbfgs_direction(grad, s_list, y_list):
    """
    Compute the L-BFGS search direction using the two-loop recursion.
    """
    # Write code here
    m = len(s_list)
    q = grad[:]

    rho = [0.0] * m
    alpha = [0.0] * m
    
    for i in range(m):
        rho[i] = 1.0 / _dot(y_list[i], s_list[i])

    for i in range(m - 1, -1, -1):
        alpha[i] = rho[i] * _dot(s_list[i], q)
        q = [qi - alpha[i] * yi for qi, yi in zip(q, y_list[i])]

    sTy = _dot(s_list[-1], y_list[-1])
    yTy = _dot(y_list[-1], y_list[-1])
    gamma = sTy / yTy
    r = [gamma * qi for qi in q]

    for i in range(m):
        beta = rho[i] * _dot(y_list[i], r)
        r = [rj + s_list[i][j] * (alpha[i] - beta) for j, rj in enumerate(r)]

    return [-rj for rj in r]