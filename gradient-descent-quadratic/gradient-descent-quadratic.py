def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = x0 - lr * (2*a*x0 + b) * x0
    
    for _ in range(steps):
        f_x = 2 * a * x + b
        x = x - lr * f_x

    return x
    