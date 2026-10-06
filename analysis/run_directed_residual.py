"""Fixed-setting diagnostic of uniform-position-corrected Directed STAS.

Uses only the existing tuning validation sample. The original runners are unchanged.
Subtract the uniform, zero-diagonal, row-normalized graph's directed degree before
standardization. This removes the uniform opportunity-count bonus, not all position bias.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from multiprocessing import Pool
from unittest.mock import patch

import numpy as np

from analysis import directed, experiment, positions, selection, stas, stats
from analysis.pmi import standardize
from analysis.rouge import METRICS, summarize

LAM_B = -0.5
MU = 1.0
SALIENCE_SETTING = 9


def uniform_degree(n):
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    i = np.arange(n, dtype=np.float64)
    return ((n - 1 - i) + LAM_B * i) / (n - 1)


def residual_degree(graph):
    if len(graph) < 2:
        return np.zeros(len(graph), dtype=np.float64)
    residual = directed.directed_degree(graph.astype(np.float64), 1.0, LAM_B) - uniform_degree(len(graph))
    if np.all(np.abs(residual) <= 1e-12):
        return np.zeros_like(residual)
    return residual


def build_settings(salience, graphs, articles):
    if not len(articles) == len(salience) == len(graphs):
        raise ValueError('scores, graphs and articles differ in length')
    settings = {name: [] for name in ('stas', 'original', 'residual')}
    for doc_id, (s, g, art) in enumerate(zip(salience, graphs, articles)):
        n = len(s)
        if n < 1 or g.shape != (n, n) or n > len(art) or not np.isfinite(s).all():
            raise ValueError(f'doc {doc_id}: invalid score/graph/text alignment')
        if n > 1 and (not np.isfinite(g).all()
                      or not np.allclose(np.diag(g), 0, atol=1e-7)
                      or not np.allclose(g.sum(axis=1), 1, atol=2e-3, rtol=0)):
            raise ValueError(f'doc {doc_id}: invalid attention graph')
        raw = directed.directed_degree(g, 1.0, LAM_B) if n > 1 else np.zeros(n)
        scores = {'stas': s,
                  'original': standardize(s) + MU * standardize(raw),
                  'residual': standardize(s) + MU * standardize(residual_degree(g))}
        for name, values in scores.items():
            settings[name].append(selection.topk_trigram_blocking(values, art))
    return settings


def score_without_settings_write(work):
    # ROUGE_HOME is supplied explicitly by analysis.rouge. Pyrouge need not persist
    # that path in ~/.pyrouge/settings.ini, which may be read-only in an agent session.
    with patch('pyrouge.Rouge155.save_home_dir', return_value=None):
        return experiment._score(work)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dump-dir', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--previous-report', required=True, help='original directed_tuning.json for reproduction gate')
    p.add_argument('--jobs', type=int, default=3)
    a = p.parse_args()
    out = Path(a.out)
    if (out / 'report.json').exists():
        p.error('report already exists; use a new output directory to preserve results')
    articles, references = experiment.load_split(a.data, 'valid')
    if len(articles) != 2000 or len(references) != len(articles):
        raise ValueError('expected the pre-registered 2,000-document tuning sample')
    sal = stas.load_output(Path(a.dump_dir) / f'{SALIENCE_SETTING}.valid.txt')
    graphs = directed.load_graphs(Path(a.dump_dir) / 'stas_dump.valid.npz')
    settings = build_settings(sal, graphs, articles)
    print('| integrity checks passed; evaluating 3 fixed systems on tuning validation', flush=True)
    names = list(settings)
    work = [(settings[name], articles, references) for name in names]
    if a.jobs > 1:
        with Pool(min(a.jobs, len(names))) as pool:
            scored = pool.map(score_without_settings_write, work)
    else:
        scored = [score_without_settings_write(w) for w in work]
    scores = dict(zip(names, scored))
    table = {name: summarize(sc) for name, sc in scores.items()}
    previous = json.loads(Path(a.previous_report).read_text())
    expected = {'stas': previous['table']['baseline: STAS + trigram blocking'],
                'original': previous['table']['directed, lam_f=1.0, lam_b=-0.5, mu=1.0']}
    reproduced = all(abs(table[name][m] - expected[name][m]) <= 1e-6
                     for name in expected for m in METRICS)
    comparisons = {}
    for x, y in (('residual', 'original'), ('residual', 'stas'), ('original', 'stas')):
        comparisons[f'{x}_minus_{y}'] = {
            m: asdict(stats.paired(scores[x][m], scores[y][m])) for m in METRICS}
        mean_x = sum(scores[x][m] for m in METRICS) / 3
        mean_y = sum(scores[y][m] for m in METRICS) / 3
        comparisons[f'{x}_minus_{y}']['mean_rouge'] = asdict(stats.paired(mean_x, mean_y))
    pos = {name: positions.distribution(sel).tolist() for name, sel in settings.items()}
    out.mkdir(parents=True, exist_ok=True)
    for name, sc in scores.items():
        experiment.save(str(out), name, 'valid', 'fixed', sc, settings[name])
    report = {'agent': 'Codex', 'protocol': 'checkpoints/v0.16.md', 'split': 'valid',
              'n_docs': len(articles), 'diagnostic_only': True, 'tuned': False,
              'fixed': {'salience_setting': SALIENCE_SETTING, 'lam_f': 1.0, 'lam_b': LAM_B, 'mu': MU},
              'reproduction_passed': reproduced, 'table': table,
              'comparisons': comparisons, 'positions': pos}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'| reproduction gate: {"PASS" if reproduced else "FAIL"}', flush=True)
    for name, t in table.items():
        print(f'| {name}: mean {experiment.mean_rouge(t):.4f}, '
              + ' / '.join(f'{t[m]:.4f}' for m in METRICS), flush=True)
    for pair, values in comparisons.items():
        print(f'| {pair}, mean ROUGE: {stats.Comparison(**values["mean_rouge"])}', flush=True)
    for name, values in pos.items():
        print(f'| {name}: first-three position share {sum(values[:3]):.4f}', flush=True)
    if not reproduced:
        raise RuntimeError('original baselines did not reproduce; do not interpret the residual result')


if __name__ == '__main__':
    main()
