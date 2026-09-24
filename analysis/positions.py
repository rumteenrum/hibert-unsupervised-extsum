"""Where in the document selected sentences come from.

STAS (Xu et al., 2020, Fig. 4) compares the position distribution of each system's
selections with that of the oracle, using KL(system || oracle).
"""

import numpy as np


def distribution(selections, n_positions=30):
    counts = np.zeros(n_positions)
    for sel in selections:
        for i in sel:
            if i < n_positions:
                counts[i] += 1
    return counts / counts.sum()


def kl(p, q, eps=1e-6):
    p, q = np.asarray(p) + eps, np.asarray(q) + eps
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))
