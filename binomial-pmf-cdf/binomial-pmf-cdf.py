import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    # Write code here
    pmf = math.comb(n, k) * p ** k * (1.0 - p) ** (n - k)
    cmf = 0

    for i in range(k + 1):
        cmf += math.comb(n, i) * p ** i * (1.0 - p) ** (n - i)

    return {"pmf": pmf, "cdf": cmf}