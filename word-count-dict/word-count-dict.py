def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    # Your code here
    cnt = {}
    
    for line in sentences:
        for word in line:
            if cnt.get(word, None) is None:
                cnt[word] = 1
            else:
                cnt[word] += 1
    
    return cnt