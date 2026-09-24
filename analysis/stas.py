"""STAS sentence ranking (Xu et al., 2020) and its output files.

`rank` reimplements the ranking step of the authors' model
(fairseq/models/extract_sum_recovery_transformer_pagerank.py in github.com/xssstory/STAS) on
precomputed model outputs, so that it can be applied to any hierarchical model:

  1. r~_i  mean probability of the true tokens of sentence i, predicted with sentence i
           masked; normalised to sum to 1 over the document.
  2. G     sentence-level attention averaged over heads and layers, diagonal set to 0,
           rows renormalised. Row i comes from the pass in which sentence i is masked
           ("multi-graph", the authors' CNN/DailyMail setting).
  3. p <- normalise(p * lambda1 / n + (p @ G) * lambda2), starting from p = r~, where
           lambda2 = 1 - lambda1. Since (p @ G)_i = sum_j p_j G[j, i], a sentence gains
           score from the sentences that attend to it, weighted by their own scores.

The authors keep four results per lambda1: after 1, 2 and 3 updates, and at convergence
(change below 1e-5, at most 21 updates). With lambda1 in {0.80, 0.84, 0.88, 0.92, 0.96}
this gives 20 settings, numbered 4 * lambda_index + slot as in their output files.
"""

import numpy as np

LAMBDAS = (0.80, 0.84, 0.88, 0.92, 0.96)
SLOTS = 4  # after 1, 2, 3 updates, and converged


def setting_name(k):
    lam, slot = LAMBDAS[k // SLOTS], k % SLOTS
    return f'lambda1={lam:.2f}, ' + ('converged' if slot == SLOTS - 1 else f'{slot + 1} update' + 's' * (slot > 0))


def sentence_recovery(token_logprobs):
    """r~: mean true-token probability per sentence, normalised over the document."""
    r = np.array([np.exp(lp.astype(np.float64)).mean() for lp in token_logprobs])
    return r / r.sum()


def attention_graph(attn):
    """G from per-layer attention [layers, n, n] (heads already averaged)."""
    g = attn.astype(np.float64).mean(axis=0)
    np.fill_diagonal(g, 0.0)
    return g / g.sum(axis=-1, keepdims=True)


def rank(recovery, graph):
    """All 20 STAS score vectors for one document, indexed like the authors' files."""
    n = len(recovery)
    if n == 1:
        return [np.ones(1)] * (len(LAMBDAS) * SLOTS)
    out = []
    for lam1 in LAMBDAS:
        lam2 = 1 - lam1
        pr, slots, step = recovery.copy(), [], 0
        while True:
            new = pr * lam1 / n + (pr @ graph) * lam2
            new = new / new.sum()
            if step < SLOTS - 1:
                slots.append(new)
            elif np.abs(pr - new).sum() < 1e-5 or step > 20:
                break
            step += 1
            pr = new
        slots.append(new)
        out.extend(slots)
    return out


def load_output(path):
    """Per-document sentence scores from one of the authors' `{k}.{split}.txt` files."""
    docs = []
    with open(path, encoding='utf8') as f:
        n_dict = int(f.readline())
        for _ in range(n_dict):
            f.readline()
        for line in f:
            if line.startswith('Score:\t'):
                docs.append(np.array([float(x) for x in line.split('\t', 1)[1].split()]))
    return docs
