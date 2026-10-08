"""Exploratory dense direction grid on a pre-registered tuning subset only."""

import argparse
from dataclasses import asdict
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from analysis import directed, experiment, positions, selection, stas, stats
from analysis.pmi import standardize
from analysis.rouge import METRICS, summarize
from analysis.run_directed import MAIN_LAM_B, MU
from analysis.run_directed_test import fixed_selections, score_fixed


def name(lb, mu):
    return f'lambda_b={lb:g}, mu={mu:g}'


def pack_unique(settings, articles, references):
    """Keep document identity and selection order in each cache key."""
    keys, index, mapping = [], {}, {}
    for setting, selections in settings.items():
        if len(selections) != len(articles) or len(references) != len(articles):
            raise ValueError('selection/text counts differ')
        ids = []
        for doc, sel in enumerate(selections):
            key = (doc, tuple(int(i) for i in sel))
            if key not in index:
                index[key] = len(keys)
                keys.append(key)
            ids.append(index[key])
        mapping[setting] = np.array(ids)
    summaries = [[articles[doc][i] for i in sel] for doc, sel in keys]
    refs = [references[doc] for doc, _ in keys]
    return summaries, refs, mapping


def batch_work(summaries, refs):
    return ([list(range(len(s))) for s in summaries], summaries, refs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, help='shared project root')
    p.add_argument('--out', required=True)
    p.add_argument('--jobs', type=int, default=3)
    a = p.parse_args()
    root, out = Path(a.root), Path(a.out)
    if (out / 'report.json').exists():
        p.error('report exists; preserve previous results')
    manifest = json.loads((out / 'manifest.json').read_text())
    if manifest['split'] != 'valid' or manifest['n_docs'] != 500:
        raise ValueError('unexpected pre-registered split/sample')
    for path, sha in manifest['source_sha256'].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != sha:
            raise ValueError(f'input hash differs: {path}')
    ids = np.array(manifest['source_indices'])
    expected = np.sort(np.random.default_rng(manifest['seed']).choice(2000, 500, replace=False))
    np.testing.assert_array_equal(ids, expected)
    articles, refs = experiment.load_split(root / 'gpu-bundle/local_eval', 'valid')
    if len(articles) != 2000 or len(refs) != 2000:
        raise ValueError('expected original 2,000 tuning documents')
    sal = stas.load_output(root / 'gpu-bundle/outputs/stas_dump/9.valid.txt')
    graphs = directed.load_graphs(root / 'gpu-bundle/outputs/stas_dump/stas_dump.valid.npz')
    baseline, original = fixed_selections(sal, graphs, articles)
    cached = {}
    for key, selections in [('stas', baseline), ('original', original)]:
        path = root / f'redundancy-extractive/results/directed_residual_tuning/{key}.valid.fixed.npz'
        with np.load(path, allow_pickle=True) as d:
            cached[key] = {m: d[m][ids] for m in METRICS}
            if selections != [list(s) for s in d['selections']]:
                raise ValueError(f'{key}: original validation selections differ')
    arts, references = [articles[i] for i in ids], [refs[i] for i in ids]
    settings, params = {'baseline': [baseline[i] for i in ids]}, {}
    zsal = [standardize(sal[i]) for i in ids]
    for lb in manifest['lambda_b']:
        zd = [standardize(directed.directed_degree(graphs[i], 1, lb)) for i in ids]
        for mu in manifest['mu']:
            key = name(lb, mu)
            params[key] = {'lambda_b': lb, 'mu': mu}
            settings[key] = [selection.topk_trigram_blocking(s + mu * d, art)
                             for s, d, art in zip(zsal, zd, arts)]
    if len(params) != 169 or settings[name(-.5, 1)] != [original[i] for i in ids]:
        raise ValueError('grid count or original selection reproduction failed')
    summaries, unique_refs, mapping = pack_unique(settings, arts, references)
    np.savez_compressed(out / 'selections.npz', source_indices=ids,
                        **{key: np.array(v, dtype=object) for key, v in settings.items()})
    chunks = [batch_work(summaries[i:i+500], unique_refs[i:i+500])
              for i in range(0, len(summaries), 500)]
    print(f'| Gates PASS; 169 settings + baseline; {len(summaries)} unique summaries '
          f'in {len(chunks)} batches instead of {len(settings)*len(ids)} evaluations', flush=True)
    results = []
    with Pool(min(max(a.jobs, 1), len(chunks))) as pool:
        for result in pool.imap(score_fixed, chunks):
            results.append(result)
            print(f'| ROUGE batch {len(results)}/{len(chunks)} complete', flush=True)
    unique = {m: np.concatenate([r[m] for r in results]) for m in METRICS}
    np.savez_compressed(out / 'unique_scores.npz', **unique)
    scores = {key: {m: unique[m][idx] for m in METRICS} for key, idx in mapping.items()}
    for key, setting in [('stas', 'baseline'), ('original', name(-.5, 1))]:
        for m in METRICS:
            np.testing.assert_allclose(scores[setting][m], cached[key][m], atol=1e-12, rtol=0)
    table = {key: summarize(v) for key, v in scores.items()}
    ranked = sorted(params, key=lambda key: -experiment.mean_rouge(table[key]))
    old_grid = [name(lb, mu) for lb in MAIN_LAM_B for mu in MU]
    old_best = max(old_grid, key=lambda key: experiment.mean_rouge(table[key]))
    best, fixed = ranked[0], name(-.5, 1)
    means = {key: sum(v[m] for m in METRICS) / 3 for key, v in scores.items()}
    comparisons = {f'{x} minus {y}': asdict(stats.paired(means[x], means[y]))
                   for x, y in [(best, fixed), (best, 'baseline'), (fixed, 'baseline')]}
    report = {'agent': 'Codex', 'protocol': 'checkpoints/v0.14.md: dense pre-registration',
              'exploratory': True, 'post_test_development': True, 'split': 'valid',
              'n_docs': len(ids), 'n_settings': len(params), 'n_unique_summaries': len(summaries),
              'reproduction_passed': True, 'table': table, 'parameters': params,
              'ranking': ranked, 'best_dense': best, 'best_original_grid_on_subset': old_best,
              'fixed_original': fixed, 'comparisons_descriptive_only': comparisons,
              'positions': {key: positions.distribution(settings[key]).tolist()
                            for key in dict.fromkeys(['baseline', fixed, best])},
              'best_on_boundary': any(params[best][key] in (manifest[key][0], manifest[key][-1])
                                      for key in ('lambda_b', 'mu'))}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(out / 'grid_scores.npz', **{f'{key}:{m}': v[m]
                                                  for key, v in scores.items() for m in METRICS})
    print('| Baseline/original per-document ROUGE reproduction PASS', flush=True)
    for key in dict.fromkeys(ranked[:10] + [fixed, old_best, 'baseline']):
        print(f'| {key}: mean {experiment.mean_rouge(table[key]):.4f}; {table[key]}', flush=True)
    for key, value in comparisons.items():
        print(f'| {key}: {stats.Comparison(**value)} (exploratory)', flush=True)


if __name__ == '__main__':
    main()
