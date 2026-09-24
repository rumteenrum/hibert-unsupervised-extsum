"""Final comparison on test: scores, paired confidence intervals, sentence positions.

Reads what the run_* scripts saved: each system's chosen setting, LEAD-3 and the oracle.
Differences are paired over documents (95% bootstrap CI, sign-flip permutation p-value).

    python -m analysis.compare --results results --systems stas hibert_stas
"""

import argparse
import glob
import json
import os

import numpy as np

from analysis import experiment, positions, stats
from analysis.rouge import METRICS


def load(results, system, split):
    if system in ('lead3', 'oracle'):
        name = system
    else:
        with open(os.path.join(results, f'{system}.json')) as f:
            name = json.load(f)['chosen']
    d = np.load(os.path.join(results, f'{system}.{split}.{experiment.file_safe(name)}.npz'), allow_pickle=True)
    return name, {m: d[m] for m in METRICS}, list(d['selections'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True)
    p.add_argument('--systems', nargs='+', default=None, help='default: every system with a <name>.json')
    p.add_argument('--split', default='test')
    a = p.parse_args()

    systems = a.systems or sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(a.results, '*.json'))
                                  if not f.endswith('baselines.json'))
    loaded = {s: load(a.results, s, a.split) for s in ['lead3', 'oracle'] + systems}
    n = min(len(v[1]['rouge_1']) for v in loaded.values())
    if any(len(v[1]['rouge_1']) != n for v in loaded.values()):
        print(f'| note: comparing on the first {n} documents only')
    loaded = {s: (name, {m: v[:n] for m, v in sc.items()}, sel[:n]) for s, (name, sc, sel) in loaded.items()}

    print(f'\n{a.split}, {n} documents\n')
    print(f'{"system":14s} {"R-1":>6s} {"R-2":>6s} {"R-L":>6s}   setting')
    for s, (name, sc, _) in loaded.items():
        print(f'{s:14s} ' + ' '.join(f'{100 * sc[m].mean():6.2f}' for m in METRICS) + f'   {name}')

    pairs = [(s, 'lead3') for s in systems]
    pairs += [(systems[i], systems[j]) for i in range(len(systems)) for j in range(i + 1, len(systems))]
    print('\npaired differences (ROUGE points, 95% CI, p)')
    for x, y in pairs:
        print(f'{x} - {y}')
        for m in METRICS:
            print(f'    {m}: {stats.paired(loaded[x][1][m], loaded[y][1][m])}')

    oracle_pos = positions.distribution(loaded['oracle'][2])
    print('\nselected sentence positions (share of picks at 0 / 1 / 2 / 3+, KL to oracle)')
    for s, (_, _, sel) in loaded.items():
        d = positions.distribution(sel)
        print(f'{s:14s} {d[0]:.2f} / {d[1]:.2f} / {d[2]:.2f} / {d[3:].sum():.2f}   KL {positions.kl(d, oracle_pos):.3f}')


if __name__ == '__main__':
    main()
