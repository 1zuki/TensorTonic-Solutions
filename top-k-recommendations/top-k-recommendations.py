def top_k_recommendations(scores, rated_indices, k):
    """
    Return indices of top-k unrated items by predicted score.
    """
    items = [(scores[i], i) for i in range(len(scores)) if i not in rated_indices]
    items.sort(key=lambda x: -x[0])
    return [i for _, i in items[:k]]