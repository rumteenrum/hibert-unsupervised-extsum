"""Tune a linear position control, then compare locked settings on a fresh holdout.

This runner has no test-set mode. Use --stage tune on the original validation
sample, then --stage holdout with new STAS outputs and the locked tuning report.
"""

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
from analysis.run_directed_residual import build_settings as reference_settings, score_without_settings_write

MU = (.25, .5, 1.0, 2.0, 4.0)


def position_scores(salience, mu):
    return standardize(salience) + mu * standardize(-np.arange(len(salience), dtype=float))


def build_settings(salience, graphs, articles, weights):
    original = reference_settings(salience, graphs, articles)
    settings = {name: original[name] for name in ('stas', 'original')}
    for mu in weights:
        settings[f'position_mu={mu}'] = [
            selection.topk_trigram_blocking(position_scores(s, mu), art)
            for s, art in zip(salience, articles)]
    return settings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=('tune', 'holdout'))
    p.add_argument('--dump-dir', required=True)
    p.add_argument('--data', required=True, help='validation sample: valid.article and valid.summary')
    p.add_argument('--out', required=True)
    p.add_argument('--previous-report', help='original directed_tuning.json (required for tune)')
    p.add_argument('--locked-report', help='this runner tuning report.json (required for holdout)')
    p.add_argument('--manifest', help='fresh holdout manifest.json (required for holdout)')
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()
    out = Path(a.out)
    if (out / 'report.json').exists():
        p.error('report already exists; preserve it and choose a new output directory')
    if a.stage == 'tune':
        if not a.previous_report:
            p.error('tune requires --previous-report')
        weights = MU
    else:
        if not a.locked_report or not a.manifest:
            p.error('holdout requires --locked-report and --manifest')
        locked = json.loads(Path(a.locked_report).read_text())
        if locked['stage'] != 'tune' or not locked['reproduction_passed']:
            raise ValueError('invalid locked tuning report')
        weights = (locked['chosen_mu'],)
        manifest_path = Path(a.manifest)
        manifest = json.loads(manifest_path.read_text())
        if manifest['seed'] != 3 or manifest['n_docs'] != 2000 or manifest['overlap_with_used_samples'] != 0:
            raise ValueError('invalid fresh holdout manifest')
        indices = {int(i) for i in (manifest_path.parent / 'indices.txt').read_text().splitlines()}
        excluded = {int(i) for i in (manifest_path.parent / 'excluded_indices.txt').read_text().splitlines()}
        if len(indices) != 2000 or len(excluded) != 4000 or indices & excluded:
            raise ValueError('fresh holdout overlaps used samples')
        for name in ('indices.txt', 'excluded_indices.txt'):
            if hashlib.sha256((manifest_path.parent / name).read_bytes()).hexdigest() != manifest['files'][name]:
                raise ValueError(f'holdout index hash mismatch: {name}')
        for ext in ('article', 'summary'):
            path = Path(a.data) / f'valid.{ext}'
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['files'][f'local_eval/valid.{ext}']:
                raise ValueError(f'holdout text hash mismatch: {ext}')
        dumped_manifest = json.loads((Path(a.dump_dir) / 'manifest.json').read_text())
        if dumped_manifest != manifest:
            raise ValueError('dump provenance differs from the holdout manifest')
    articles, references = experiment.load_split(a.data, 'valid')
    if len(articles) != 2000 or len(references) != len(articles):
        raise ValueError('expected the pre-registered 2,000-document validation sample')
    sal = stas.load_output(Path(a.dump_dir) / '9.valid.txt')
    graphs = directed.load_graphs(Path(a.dump_dir) / 'stas_dump.valid.npz')
    if len(sal) != len(graphs):
        raise ValueError('score and graph document counts differ')
    # The authors' output may append -inf score slots when a document has <3 sentences.
    for i, (s, g) in enumerate(zip(sal, graphs)):
        n = len(g)
        if len(s) > n:
            if n >= 3 or not np.isneginf(s[n:]).all():
                raise ValueError(f'doc {i}: unexpected extra scores')
            sal[i] = s[:n]
    settings = build_settings(sal, graphs, articles, weights)
    print(f'| {a.stage}: integrity passed; scoring {len(settings)} systems', flush=True)
    names = list(settings)
    work = [(settings[name], articles, references) for name in names]
    with Pool(min(max(1, a.jobs), len(names))) as pool:
        scores = dict(zip(names, pool.map(score_without_settings_write, work)))
    table = {name: summarize(sc) for name, sc in scores.items()}
    reproduction = None
    if a.stage == 'tune':
        previous = json.loads(Path(a.previous_report).read_text())['table']
        expected = {'stas': previous['baseline: STAS + trigram blocking'],
                    'original': previous['directed, lam_f=1.0, lam_b=-0.5, mu=1.0']}
        reproduction = all(abs(table[name][m] - expected[name][m]) <= 1e-6
                           for name in expected for m in METRICS)
        if not reproduction:
            raise RuntimeError('STAS/original reproduction failed; do not choose a control setting')
        chosen_mu = max(MU, key=lambda mu: experiment.mean_rouge(table[f'position_mu={mu}']))
    else:
        chosen_mu = weights[0]
    chosen = f'position_mu={chosen_mu}'
    comparisons = {}
    for x, y in (('original', chosen), (chosen, 'stas'), ('original', 'stas')):
        values = {m: asdict(stats.paired(scores[x][m], scores[y][m])) for m in METRICS}
        values['mean_rouge'] = asdict(stats.paired(
            sum(scores[x][m] for m in METRICS) / 3,
            sum(scores[y][m] for m in METRICS) / 3))
        comparisons[f'{x}_minus_{y}'] = values
    pos = {name: positions.distribution(sel).tolist() for name, sel in settings.items()}
    report = {'agent': 'Codex', 'protocol': 'checkpoints/v0.17.md', 'stage': a.stage,
              'n_docs': len(articles), 'grid': list(weights), 'chosen_mu': chosen_mu,
              'chosen': chosen, 'reproduction_passed': reproduction,
              'diagnostic_only': a.stage == 'tune', 'table': table,
              'comparisons': comparisons, 'positions': pos}
    if a.stage == 'holdout':
        report['manifest'] = manifest
        report['locked_report_sha256'] = hashlib.sha256(Path(a.locked_report).read_bytes()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    for name, sc in scores.items():
        experiment.save(str(out), name, 'valid', 'fixed', sc, settings[name])
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    for name, t in table.items():
        print(f'| {name}: mean {experiment.mean_rouge(t):.4f}, '
              + ' / '.join(f'{t[m]:.4f}' for m in METRICS), flush=True)
    print(f'| locked position weight: {chosen_mu}', flush=True)
    for pair, values in comparisons.items():
        print(f'| {pair}, mean ROUGE: {stats.Comparison(**values["mean_rouge"])}', flush=True)
    for name in ('stas', 'original', chosen):
        print(f'| {name}: first-three position share {sum(pos[name][:3]):.4f}', flush=True)


if __name__ == '__main__':
    main()
