"""Coverage-based sentence selection with masked-sentence PMI.

Choose the summary S that best explains the rest of the document, plus a salience bonus:

    F(S) = (1/n) * sum_d max_{s in S} C[s, d]  +  mu * sum_{s in S} z(salience(s))

C[s, d] is how much sentence s helps the model predict sentence d (pairwise PMI from
hibert_pairs.py, clipped at 0). Only "earlier explains later" is measured directly; the reverse
direction enters with weight rho. The first term is a facility-location function, so F is
monotone submodular for mu >= 0 and greedy selection is within (1 - 1/e) of the optimum. Unlike a
redundancy penalty, the max only discounts content that a chosen sentence already explains.
"""

import numpy as np

from analysis.selection import trigrams


def coverage_matrix(pair_rec, rho):
    """C[s, d] for the document's first n sentences."""
    n = pair_rec['n']
    pair = pair_rec['pair_mean_logprob'].astype(np.float64)
    base = pair_rec['base_mean_logprob'].astype(np.float64)
    c = np.zeros((n, n))
    iu = np.triu_indices(n, 1)                       # s < d: s explains the later d
    forward = np.maximum(0.0, pair[iu] - base[iu[1]])
    c[iu] = forward
    c[iu[1], iu[0]] = rho * forward                  # s > d: the same pair, other direction
    np.fill_diagonal(c, c.max() if n > 1 else 1.0)   # a sentence covers itself fully
    return c


def objective(c, chosen, salience, mu):
    if not chosen:
        return 0.0
    return c[chosen].max(axis=0).mean() + mu * salience[chosen].sum()


def greedy(c, salience, mu, candidates, k=3, sentences=None):
    """Greedy maximisation of F over the first `candidates` sentences.

    sentences: when given, trigram blocking is applied as in STAS's evaluation code.
    Returns the chosen indices in the order chosen."""
    covered = np.zeros(c.shape[1])
    chosen, seen = [], set()
    remaining = list(range(candidates))
    while remaining and len(chosen) < k:
        gains = [(np.maximum(0.0, c[s] - covered).mean() + mu * salience[s], s) for s in remaining]
        for _, s in sorted(gains, key=lambda g: (-g[0], g[1])):
            if sentences is not None:
                tri = trigrams(sentences[s])
                if any(t in seen for t in tri):
                    remaining.remove(s)
                    continue
                seen.update(tri)
            chosen.append(s)
            covered = np.maximum(covered, c[s])
            remaining.remove(s)
            break
        else:
            break
    return chosen
