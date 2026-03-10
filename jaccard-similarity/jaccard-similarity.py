def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here
    if len(set_a) == 0 and len(set_b) == 0:
        return 0.0

    set_a = set(set_a)
    set_b = set(set_b)

    return len(set_a & set_b) / len(set_a | set_b)