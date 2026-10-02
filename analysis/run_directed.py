"""Screen Directed STAS on a validation sample only.

Combines STAS's score with the directed attention degree from its own graph,
z(STAS) + mu * z(D), and compares every setting of a fixed grid with STAS (same setting, trigram
blocking) computed in the same run. The test set is never read.

    python -m analysis.run_directed --dump-dir OUT/stas_dump --data DIR --out results
"""

import argparse
import json
import os

import numpy as np

from analysis import directed, experiment, positions, selection, stas, stats
from analysis.pmi import standardize
from analysis.rouge import summarize
from analysis.run_select import SALIENCE_SETTING

MU = (0.25, 0.5, 1.0, 2.0, 4.0)
MAIN_LAM_B = (0.0, -0.5, -1.0, -2.0)
ABLATIONS = {'undirected': (1.0, 1.0), 'backward only': (0.0, 1.0)}
BASELINE = 'baseline: STAS + trigram blocking'


def build_settings(dump_dir, articles, split='valid'):
    sal = stas.load_output(os.path.join(dump_dir, f'{SALIENCE_SETTING}.{split}.txt'))
    graphs = directed.load_graphs(os.path.join(dump_dir, f'stas_dump.{split}.npz'))
    if len(sal) != len(graphs) or len(sal) != len(articles):
        raise ValueError('scores, graphs and articles differ in length')
    for s, g in zip(sal, graphs):
        if len(s) != g.shape[0]:
            raise ValueError('score vector and graph size differ')

    zsal = [standardize(s) for s in sal]
    pick = lambda scores: [selection.topk_trigram_blocking(sc, a) for sc, a in zip(scores, articles)]
    settings = {BASELINE: pick(sal)}

    families = [('directed', 1.0, lb) for lb in MAIN_LAM_B] + [(name, lf, lb) for name, (lf, lb) in ABLATIONS.items()]
    for name, lf, lb in families:
        zd = [standardize(directed.directed_degree(g, lf, lb)) for g in graphs]
        if name == 'directed':
            settings[f'degree only, lam_b={lb}'] = pick(zd)
        for mu in MU:
            settings[f'{name}, lam_f={lf}, lam_b={lb}, mu={mu}'] = pick([s + mu * d for s, d in zip(zsal, zd)])
    return settings


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dump-dir', required=True, help='patched STAS output dir (score files + stas_dump npz)')
    p.add_argument('--data', required=True)
    p.add_argument('--split-name', default='valid', help='name of the text files in --data (valid or valid2)')
    p.add_argument('--out', required=True)
    p.add_argument('--tag', default='tuning')
    p.add_argument('--only', default=None, help='score only this setting (plus the baseline), e.g. for confirmation')
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()

    articles, references = experiment.load_split(a.data, a.split_name)
    settings = build_settings(a.dump_dir, articles)
    if a.only:
        settings = {n: settings[n] for n in (BASELINE, a.only)}
    scores = experiment.evaluate(settings, articles, references, a.jobs)
    table = {n: summarize(s) for n, s in scores.items()}

    main_names = [n for n in table if n.startswith('directed,')]  # with --only: just the fixed setting
    best = max(main_names, key=lambda n: experiment.mean_rouge(table[n]))
    per_doc = {n: (scores[n]['rouge_1'] + scores[n]['rouge_2'] + scores[n]['rouge_l']) / 3 for n in (best, BASELINE)}
    gain = stats.paired(per_doc[best], per_doc[BASELINE])
    pos = {n: positions.distribution(settings[n]).round(3).tolist()[:4] for n in (best, BASELINE)}

    report = {'tag': a.tag, 'n_docs': len(articles), 'table': table, 'best_main': best,
              'best_minus_baseline_mean_rouge': str(gain), 'positions_first4': pos}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f'directed_{a.tag}.json'), 'w') as f:
        json.dump(report, f, indent=2)

    ranked = sorted(table, key=lambda n: -experiment.mean_rouge(table[n]))
    print(f'| Directed STAS screen ({a.tag}, {len(articles)} docs)')
    for n in ranked[:8] + ([BASELINE] if BASELINE not in ranked[:8] else []):
        t = table[n]
        print(f'|   {experiment.mean_rouge(t):6.2f}  ({t["rouge_1"]:.2f} / {t["rouge_2"]:.2f} / {t["rouge_l"]:.2f})  {n}')
    for name in ('undirected', 'backward only'):
        cands = [k for k in table if k.startswith(name)]
        if cands:
            n = max(cands, key=lambda k: experiment.mean_rouge(table[k]))
            print(f'|   ablation best {name}: {experiment.mean_rouge(table[n]):.2f}  ({n})')
    print(f'| best main − baseline (mean ROUGE, paired 95% CI): {gain}')
    print(f'| positions (share at 0/1/2/3): best {pos[best]}  baseline {pos[BASELINE]}')


if __name__ == '__main__':
    main()
