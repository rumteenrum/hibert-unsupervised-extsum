"""Paired comparisons of per-document scores.

Both systems must be scored on the same documents in the same order. Differences are
reported as system a minus system b, in ROUGE points (0-100).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Comparison:
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float

    def __str__(self):
        return f'{self.mean_diff:+.2f} [{self.ci_low:+.2f}, {self.ci_high:+.2f}] p={self.p_value:.3g}'


def paired(a, b, n_boot=10_000, n_perm=10_000, seed=0):
    """95% bootstrap CI of mean(a - b) and a two-sided sign-flip permutation p-value."""
    d = 100 * (np.asarray(a, float) - np.asarray(b, float))
    if d.ndim != 1 or len(d) == 0:
        raise ValueError('need two equal-length 1-D score arrays')
    rng = np.random.default_rng(seed)

    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boot, [0.025, 0.975])

    observed = abs(d.mean())
    null = np.array([abs((d * rng.choice((-1.0, 1.0), len(d))).mean()) for _ in range(n_perm)])
    p = (np.sum(null >= observed - 1e-12) + 1) / (n_perm + 1)
    return Comparison(float(d.mean()), float(lo), float(hi), float(p))
