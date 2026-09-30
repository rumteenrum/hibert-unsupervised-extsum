"""Screen coverage selection (analysis.coverage) on the validation sample only.

Scores every setting of a fixed grid against the salience-only baseline with trigram blocking,
computed in the same run, and reports the best setting with a paired 95% CI of its gain in mean
ROUGE-1/2/L. The test set is never read.

    python -m analysis.run_coverage --salience stas --stas-dir OUT/stas --hibert-dir OUT/hibert \
        --pairs-dir OUT/pairs --data DIR --out results
"""

import argparse
import json
import os

import numpy as np

from analysis import coverage, experiment, hibert, pmi, selection, stats
from analysis.rouge import summarize
from analysis.run_select import salience_for

RHO = (0.0, 0.5, 1.0)
MU = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
BASELINE = 'baseline: salience + trigram blocking'


def build_settings(kind, stas_dir, hibert_dir, pairs_dir, articles):
    records = hibert.load_split(os.path.join(hibert_dir, 'valid'))[:len(articles)]
    pairs = hibert.load_split(os.path.join(pairs_dir, 'valid'))[:len(articles)]
    sal = salience_for(kind, 'valid', stas_dir, records)

    settings = {BASELINE: [selection.topk_trigram_blocking(s, a) for s, a in zip(sal, articles)]}
    for rho in RHO:
        mats = [coverage.coverage_matrix(p, rho) for p in pairs]
        for mu in MU:
            for block in (False, True):
                name = f'rho={rho}, mu={mu}, ' + ('trigram blocking' if block else 'no blocking')
                settings[name] = [
                    coverage.greedy(c, pmi.standardize(s), mu, candidates=len(s),
                                    sentences=a if block else None)
                    for c, s, a in zip(mats, sal, articles)]
    return settings


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--salience', choices=['stas', 'hibert'], required=True)
    p.add_argument('--stas-dir')
    p.add_argument('--hibert-dir', required=True)
    p.add_argument('--pairs-dir', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()

    n_docs = len(hibert.load_split(os.path.join(a.pairs_dir, 'valid')))
    articles, references = experiment.load_split(a.data, 'valid', n_docs)
    settings = build_settings(a.salience, a.stas_dir, a.hibert_dir, a.pairs_dir, articles)
    scores = experiment.evaluate(settings, articles, references, a.jobs)

    table = {name: summarize(s) for name, s in scores.items()}
    best = max((n for n in table if n != BASELINE), key=lambda n: experiment.mean_rouge(table[n]))
    per_doc = {n: (scores[n]['rouge_1'] + scores[n]['rouge_2'] + scores[n]['rouge_l']) / 3 for n in (best, BASELINE)}
    gain = stats.paired(per_doc[best], per_doc[BASELINE])

    report = {'salience': a.salience, 'split': 'valid', 'n_docs': len(articles), 'valid': table,
              'best': best, 'best_minus_baseline_mean_rouge': str(gain)}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f'coverage_{a.salience}.valid.json'), 'w') as f:
        json.dump(report, f, indent=2)

    ranked = sorted(table, key=lambda n: -experiment.mean_rouge(table[n]))
    print(f'| coverage screen, salience = {a.salience}, validation ({len(articles)} docs)')
    for n in ranked[:6] + ([BASELINE] if BASELINE not in ranked[:6] else []):
        t = table[n]
        print(f'|   {experiment.mean_rouge(t):6.2f}  ({t["rouge_1"]:.2f} / {t["rouge_2"]:.2f} / {t["rouge_l"]:.2f})  {n}')
    print(f'| best − baseline (mean ROUGE, paired 95% CI): {gain}')


if __name__ == '__main__':
    main()
