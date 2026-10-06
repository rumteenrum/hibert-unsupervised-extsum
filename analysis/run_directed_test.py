"""Evaluate only the locked original directed methods on saved NeuSum test outputs.

Six systems: STAS, HIBERT prefix, HIBERT full, each baseline and original direction.
No tuning or alternative methods; cached baseline selections/scores must reproduce.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis import directed, experiment, hibert, positions, selection, stas, stats
from analysis.pmi import standardize
from analysis.rouge import METRICS, summarize
from analysis.run_directed_hibert_residual import variant_inputs
from analysis.run_directed_residual import score_without_settings_write

BASELINES = {
    'stas': 'stas.test.lambda1=0.88_2_updates_trigram_blocking.npz',
    'hibert_prefix': 'hibert_stas_prefix_graph.test.lambda1=0.92_3_updates_trigram_blocking.npz',
    'hibert_full': 'hibert_stas.test.lambda1=0.88_2_updates_trigram_blocking.npz',
}


def score_fixed(work):
    # Pyrouge otherwise lists the same model directory once per document. Cache
    # only that listing after conversion; delegate regex matching to its original
    # implementation. No changes to references, ordering, evaluator or flags.
    from pyrouge import Rouge155
    original = Rouge155._Rouge155__get_model_filenames_for_id
    listdir = os.listdir
    listings = {}

    def lookup(doc_id, directory, pattern):
        if directory not in listings:
            listings[directory] = listdir(directory)
        def cached(path):
            return listings[directory] if path == directory else listdir(path)
        with patch.object(os, 'listdir', new=cached):
            return original(doc_id, directory, pattern)

    with patch.object(Rouge155, '_Rouge155__get_model_filenames_for_id', new=staticmethod(lookup)):
        return score_without_settings_write(work)


def fixed_selections(salience, graphs, articles):
    if not len(salience) == len(graphs) == len(articles):
        raise ValueError('scores, graphs and texts differ in count')
    baseline, original = [], []
    for doc_id, (s, g, art) in enumerate(zip(salience, graphs, articles)):
        n = len(g)
        if len(s) > n and n < 3 and np.isneginf(s[n:]).all():
            s = s[:n]  # authors' output padding for short documents
        if not 1 <= n <= len(art) or len(s) != n or g.shape != (n, n) or not np.isfinite(s).all():
            raise ValueError(f'doc {doc_id}: score/graph/text alignment invalid')
        if n > 1 and (not np.isfinite(g).all() or not np.allclose(np.diag(g), 0, atol=1e-7)
                      or not np.allclose(g.sum(axis=1), 1, atol=2e-3, rtol=0)):
            raise ValueError(f'doc {doc_id}: invalid graph')
        raw = directed.directed_degree(g, 1, -.5) if n > 1 else np.zeros(n)
        baseline.append(selection.topk_trigram_blocking(s, art))
        original.append(selection.topk_trigram_blocking(standardize(s) + standardize(raw), art))
    return baseline, original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dump-dir', required=True)
    p.add_argument('--hibert-dir', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--baseline-results', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--jobs', type=int, default=3)
    a = p.parse_args()
    out = Path(a.out)
    if (out / 'report.json').exists():
        p.error('report exists; preserve it and choose a new output directory')
    articles, references = experiment.load_split(a.data, 'test')
    if len(articles) != 11490 or len(references) != 11490:
        raise ValueError('expected exactly 11,490 test documents')
    dump = Path(a.dump_dir)
    salience = stas.load_output(dump / '9.test.txt')
    graphs = directed.load_graphs(dump / 'stas_dump.test.npz')
    settings = {}
    settings['stas_baseline'], settings['stas_directed'] = fixed_selections(salience, graphs, articles)
    records = hibert.load_split(str(Path(a.hibert_dir) / 'test'))
    if len(records) != 11490:
        raise ValueError('expected 11,490 HIBERT test records')
    prefixes = [len(s) for s in stas.load_output(dump / '0.test.txt')]
    for variant in ('prefix', 'full'):
        salience, graphs = variant_inputs(records, articles, prefixes, variant)
        key = f'hibert_{variant}'
        settings[f'{key}_baseline'], settings[f'{key}_directed'] = fixed_selections(salience, graphs, articles)
    cached = {}
    for key, filename in BASELINES.items():
        with np.load(Path(a.baseline_results) / filename, allow_pickle=True) as d:
            cached[key] = {k: d[k].copy() for k in d.files}
        if settings[f'{key}_baseline'] != [list(s) for s in cached[key]['selections']]:
            raise RuntimeError(f'{key}: cached baseline selections differ; abort before scoring')
    print('| All input integrity and three baseline selection gates PASS; scoring six locked systems', flush=True)
    names = list(settings)
    with Pool(min(max(a.jobs, 1), len(names))) as pool:
        scores = dict(zip(names, pool.map(score_fixed,
                                        [(settings[k], articles, references) for k in names])))
    for key in BASELINES:
        if not all(np.allclose(scores[f'{key}_baseline'][m], cached[key][m], atol=1e-12, rtol=0)
                   for m in METRICS):
            raise RuntimeError(f'{key}: cached per-document ROUGE failed reproduction')
    print('| All three per-document baseline ROUGE reproduction gates PASS', flush=True)
    table = {k: summarize(v) for k, v in scores.items()}
    comparisons = {}
    for key in BASELINES:
        x, y = scores[f'{key}_directed'], scores[f'{key}_baseline']
        values = {m: asdict(stats.paired(x[m], y[m])) for m in METRICS}
        values['mean_rouge'] = asdict(stats.paired(sum(x[m] for m in METRICS) / 3,
                                                sum(y[m] for m in METRICS) / 3))
        comparisons[key] = values
    pos = {k: positions.distribution(v).tolist() for k, v in settings.items()}
    report = {'agent': 'Codex', 'protocol': 'checkpoints/v0.14.md: original test pre-registration',
              'split': 'test', 'n_docs': 11490, 'fixed': {'lam_f': 1, 'lam_b': -.5, 'mu': 1},
              'salience_settings': {'stas': 9, 'hibert_prefix': 14, 'hibert_full': 9},
              'baseline_reproduction_passed': True, 'table': table,
              'directed_minus_baseline': comparisons, 'positions': pos,
              'text_sha256': {ext: hashlib.sha256((Path(a.data) / f'test.{ext}').read_bytes()).hexdigest()
                              for ext in ('article', 'summary')}}
    out.mkdir(parents=True, exist_ok=True)
    for key in names:
        experiment.save(str(out), key, 'test', 'fixed', scores[key], settings[key])
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    for key, t in table.items():
        print(f'| {key}: mean {experiment.mean_rouge(t):.4f}; '
              + ' / '.join(f'{t[m]:.4f}' for m in METRICS), flush=True)
    for key, c in comparisons.items():
        print(f'| {key}, directed minus baseline: {stats.Comparison(**c["mean_rouge"])}', flush=True)
    for key, v in pos.items():
        print(f'| {key}: first-three share {sum(v[:3]):.4f}', flush=True)


if __name__ == '__main__':
    main()
