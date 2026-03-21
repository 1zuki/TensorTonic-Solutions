import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # Write code here
    n = len(documents)

    if n == 0:
        return np.zeros((0, 0), dtype = float), []

    tokenized = [doc.lower().split() for doc in documents]
    vocab = sorted({token for doc in tokenized for token in doc})

    if len(vocab) == 0:
        return np.zeros((n, 0), dtype = float), []

    word_idx = {word: i for i, word in enumerate(vocab)}

    doc_freq = Counter()
    for doc in tokenized:
        for word in set(doc):
            doc_freq[word] += 1

    idf = np.zeros(len(vocab), dtype = float)
    for word, idx in word_idx.items():
        idf[idx] = math.log(n / doc_freq[word])

    tf_idf = np.zeros((n, len(vocab)), dtype = float)

    for i, doc in enumerate(tokenized):
        if len(doc) == 0:
            continue

        cnt = Counter(doc)
        total = len(doc)

        for word, count in cnt.items():
            j = word_idx[word]
            tf = count / total
            tf_idf[i, j] = tf * idf[j]

    return tf_idf, vocab
