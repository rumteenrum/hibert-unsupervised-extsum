"""PMI relevance and redundancy from HIBERT's masked-sentence decoder.

Following Padmakumar & He (2021), with probabilities from the hierarchical model instead of a
left-to-right language model. All log-probabilities are averaged over a sentence's tokens.

  relevance(s)  = log p(s | document without s) - log p(s | nothing)
                  (from hibert_extract.py: token_logprobs and uncond_logprobs)
  pmi(a; b)     = log p(b | [a, MASK(b)]) - log p(b | [empty, MASK(b)]),  a earlier than b
                  (from hibert_pairs.py); used as a symmetric redundancy between a and b,
                  conditioning on the earlier sentence as in their eq. 6.
"""

from collections import Counter

import numpy as np


def relevance(rec):
    return np.array([c.mean() - u.mean() for c, u in zip(rec['token_logprobs'], rec['uncond_logprobs'])],
                    dtype=np.float64)


def pmi_matrix(pair_rec):
    """Symmetric [n, n] redundancy matrix with zero diagonal."""
    pair = pair_rec['pair_mean_logprob'].astype(np.float64)
    base = pair_rec['base_mean_logprob'].astype(np.float64)
    n = pair_rec['n']
    m = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    m[iu] = pair[iu] - base[iu[1]]
    return m + m.T


def cosine_matrix(sentences):
    """Word-count cosine similarity between sentences (lowercased whitespace tokens)."""
    vecs = [Counter(s.lower().split()) for s in sentences]
    norms = [np.sqrt(sum(v * v for v in c.values())) for c in vecs]
    n = len(sentences)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if norms[i] and norms[j]:
                dot = sum(v * vecs[j].get(w, 0) for w, v in vecs[i].items())
                m[i, j] = m[j, i] = dot / (norms[i] * norms[j])
    return m


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)
