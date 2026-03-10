import math
from collections import Counter

def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    """
    # Write code here
    len_c = len(candidate)
    len_r = len(reference)

    if len_c == 0:
        return 0.0

    precisions = []

    for n in range(1, max_n + 1):
        cand_total = len_c - n + 1
        ref_total = len_r - n + 1

        if cand_total <= 0:
            return 0.0

        cand_ngrams = Counter(tuple(candidate[i:i+n]) for i in range(cand_total))
        ref_ngrams = Counter(tuple(reference[i:i+n]) for i in range(max(0, ref_total)))

        clipped = 0

        for ng, count in cand_ngrams.items():
            clipped += min(count, ref_ngrams.get(ng, 0))

        p_n = clipped / cand_total

        if p_n == 0:
            return 0.0

        precisions.append(p_n)

    if len_c < len_r:
        bp = math.exp(1 - len_r / len_c)

    else:
        bp = 1.0

    bleu = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    
    return float(bleu)