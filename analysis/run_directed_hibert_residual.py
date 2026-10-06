"""Fixed uniform-null correction on both existing HIBERT validation variants.

Reuses saved masked attention; no GPU inference, tuning, or test evaluation.
Original HIBERT runners and results are preserved.
"""

import argparse
from dataclasses import asdict
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from analysis import experiment, hibert, positions, stas, stats
from analysis.rouge import METRICS, summarize
from analysis.run_directed_hibert import SALIENCE_SETTING
from analysis.run_directed_residual import build_settings, score_without_settings_write
from analysis.run_hibert_stas import truncate


def variant_inputs(records, articles, prefixes, variant):
    if len(records) != len(articles) or len(prefixes) != len(articles):
        raise ValueError('record, text and prefix counts differ')
    salience, graphs = [], []
    for rec, art, prefix in zip(records, articles, prefixes):
        if rec['n_sents_total'] != len(art) or not 1 <= rec['n'] <= len(art):
            raise ValueError(f'doc {rec["doc_id"]}: HIBERT/text alignment differs')
        n = min(prefix, rec['n']) if variant == 'prefix' else rec['n']
        if n < 1:
            raise ValueError('empty candidate prefix')
        rec = truncate(rec, n)
        graph = stas.attention_graph(rec['attn_masked_rows']) if n > 1 else np.zeros((1, 1))
        recovery = stas.sentence_recovery(rec['token_logprobs'])
        salience.append(stas.rank(recovery, graph)[SALIENCE_SETTING[variant]])
        graphs.append(graph)
    return salience, graphs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hibert-dir', required=True)
    p.add_argument('--stas-dir', required=True, help='original STAS prefix score files')
    p.add_argument('--data', required=True, help='original tuning validation texts')
    p.add_argument('--previous-results', required=True, help='v0.15 HIBERT JSON directory')
    p.add_argument('--out', required=True)
    p.add_argument('--jobs', type=int, default=3)
    a = p.parse_args()
    out = Path(a.out)
    if any((out / variant / 'report.json').exists() for variant in ('prefix', 'full')):
        p.error('report exists; preserve it and choose a new output directory')
    articles, references = experiment.load_split(a.data, 'valid')
    records = hibert.load_split(str(Path(a.hibert_dir) / 'valid'))
    prefixes = [len(s) for s in stas.load_output(Path(a.stas_dir) / '0.valid.txt')]
    if len(articles) != 2000 or len(references) != 2000 or len(records) != 2000:
        raise ValueError('expected exactly the original 2,000 tuning documents')
    for variant in ('prefix', 'full'):
        salience, graphs = variant_inputs(records, articles, prefixes, variant)
        settings = build_settings(salience, graphs, articles)
        settings['baseline'] = settings.pop('stas')
        print(f'| {variant}: integrity passed; evaluating 3 fixed systems', flush=True)
        names = list(settings)
        work = [(settings[name], articles, references) for name in names]
        with Pool(min(max(a.jobs, 1), len(names))) as pool:
            scores = dict(zip(names, pool.map(score_without_settings_write, work)))
        table = {name: summarize(sc) for name, sc in scores.items()}
        previous = json.loads((Path(a.previous_results) / f'directed_hibert_{variant}.valid.json').read_text())
        prior_baselines = [name for name in previous['table'] if name.startswith('baseline:')]
        if len(prior_baselines) != 1 or previous['n_docs'] != 2000 or previous['split'] != 'valid':
            raise ValueError('invalid reference report')
        expected = {'baseline': previous['table'][prior_baselines[0]],
                    'original': previous['table']['directed, lam_f=1, lam_b=-0.5, mu=1']}
        if not all(abs(table[name][m] - expected[name][m]) <= 1e-6
                   for name in expected for m in METRICS):
            raise RuntimeError(f'{variant}: baseline/original reproduction failed')
        comparisons = {}
        for x, y in (('residual', 'original'), ('residual', 'baseline'), ('original', 'baseline')):
            values = {m: asdict(stats.paired(scores[x][m], scores[y][m])) for m in METRICS}
            values['mean_rouge'] = asdict(stats.paired(
                sum(scores[x][m] for m in METRICS) / 3,
                sum(scores[y][m] for m in METRICS) / 3))
            comparisons[f'{x}_minus_{y}'] = values
        pos = {name: positions.distribution(sel).tolist() for name, sel in settings.items()}
        report = {'agent': 'Codex', 'protocol': 'checkpoints/v0.18.md', 'variant': variant,
                  'split': 'valid', 'n_docs': 2000, 'diagnostic_only': True,
                  'salience_setting': SALIENCE_SETTING[variant], 'reproduction_passed': True,
                  'fixed': {'lam_f': 1, 'lam_b': -.5, 'mu': 1},
                  'table': table, 'comparisons': comparisons, 'positions': pos}
        dest = out / variant
        dest.mkdir(parents=True, exist_ok=True)
        for name, sc in scores.items():
            experiment.save(str(dest), name, 'valid', 'fixed', sc, settings[name])
        (dest / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'| {variant}: baseline/original reproduction PASS', flush=True)
        for name, t in table.items():
            print(f'| {name}: mean {experiment.mean_rouge(t):.4f}; '
                  + ' / '.join(f'{t[m]:.4f}' for m in METRICS), flush=True)
        for pair, values in comparisons.items():
            print(f'| {pair}, mean: {stats.Comparison(**values["mean_rouge"])}', flush=True)
        for name in settings:
            print(f'| {name}: first-three share {sum(pos[name][:3]):.4f}', flush=True)


if __name__ == '__main__':
    main()
